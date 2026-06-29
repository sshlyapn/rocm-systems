// vk_layer.cpp
//
// Vulkan/RADV counterpart of pm4_layer/hip_layer: runs the SAME all-elementwise
// dependent decode-layer chain (x += w; x *= w; x += w) N times on one compute
// queue, with a SHADER_WRITE->SHADER_READ pipeline barrier between every
// dispatch (exactly what ggml-vulkan emits between dependent ops; RADV lowers it
// to CS_PARTIAL_FLUSH + ACQUIRE_MEM full-L2 flush). Measures CPU wall time of a
// single submission of all 3*N dispatches, to compare apples-to-apples with the
// PM4 (raw KFD) and HIP/AQL numbers.
//
// Buffers x,w are DEVICE_LOCAL and host-visible (ReBAR) for init + checksum.
// Init matches the others: x[i]=(i%7)*0.01+0.1, w[i]=1+(i%5)*0.001, so the final
// checksum can be cross-checked against pm4_layer/hip_layer CHAIN=elem.

#include <vulkan/vulkan.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <vector>

#define VKCHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){ \
    fprintf(stderr,"VK error %d at %s:%d\n",(int)_r,__FILE__,__LINE__); exit(1);} } while(0)

static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

static std::vector<uint32_t> read_spv(const char* path){
    std::ifstream f(path, std::ios::binary|std::ios::ate);
    if(!f){ fprintf(stderr,"cannot open SPIR-V %s\n",path); exit(1); }
    size_t sz=(size_t)f.tellg(); std::vector<uint32_t> d(sz/4);
    f.seekg(0); f.read((char*)d.data(),sz); return d;
}
static int find_mem_type(VkPhysicalDevice pd,uint32_t bits,VkMemoryPropertyFlags want){
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)
        if((bits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&want)==want) return (int)i;
    return -1;
}

struct Buf { VkBuffer buf; VkDeviceMemory mem; void* map; };
static Buf make_buf(VkPhysicalDevice pd,VkDevice dev,VkDeviceSize sz){
    Buf b{};
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size=sz<256?256:sz; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VKCHECK(vkCreateBuffer(dev,&bci,nullptr,&b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,b.buf,&mr);
    // Prefer device-local + host-visible (ReBAR/VRAM, mappable). Fall back to host.
    int mt=find_mem_type(pd,mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if(mt<0) mt=find_mem_type(pd,mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if(mt<0){ fprintf(stderr,"no mappable mem type\n"); exit(1); }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize=mr.size; mai.memoryTypeIndex=(uint32_t)mt;
    VKCHECK(vkAllocateMemory(dev,&mai,nullptr,&b.mem));
    VKCHECK(vkBindBufferMemory(dev,b.buf,b.mem,0));
    VKCHECK(vkMapMemory(dev,b.mem,0,VK_WHOLE_SIZE,0,&b.map));
    return b;
}

int main(int argc,char** argv){
    int N=argc>1?atoi(argv[1]):1000;
    uint32_t M=argc>2?(uint32_t)atoi(argv[2]):4096;
    int barrier=argc>3?atoi(argv[3]):1;
    const char* chainEnv=getenv("CHAIN");
    bool rev = chainEnv && !strcmp(chainEnv,"rev");
    bool mix = chainEnv && !strcmp(chainEnv,"mix");
    bool serial = chainEnv && !strcmp(chainEnv,"serial");
    int  rerec = getenv("VK_RERECORD") != nullptr; // mix only: re-record cmd buffer + re-write
                                                   // descriptors every iter, like llama.cpp decode
    int  LAYERS = getenv("LAYERS") ? atoi(getenv("LAYERS")) : 1;
    if(LAYERS<1) LAYERS=1;                          // mix: repeat 5-kernel block LAYERS times/token
    int KLEN = getenv("KLEN") ? atoi(getenv("KLEN")) : 64;
    if(KLEN<2) KLEN=2; KLEN &= ~1;
    int kpi = rev ? KLEN : ((mix||serial) ? 5*LAYERS : 3);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    VkInstance instance; VKCHECK(vkCreateInstance(&ici,nullptr,&instance));
    uint32_t ndev=0; vkEnumeratePhysicalDevices(instance,&ndev,nullptr);
    std::vector<VkPhysicalDevice> devs(ndev); vkEnumeratePhysicalDevices(instance,&ndev,devs.data());
    if(!ndev){ fprintf(stderr,"no Vulkan devices\n"); exit(1); }
    VkPhysicalDevice pd=devs[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd,&props);

    uint32_t nqf=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nqf,nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf); vkGetPhysicalDeviceQueueFamilyProperties(pd,&nqf,qfs.data());
    int qf=-1,qfd=-1,qfa=-1;
    for(uint32_t i=0;i<nqf;i++){ bool c=qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT, g=qfs[i].queueFlags&VK_QUEUE_GRAPHICS_BIT;
        if(c&&qfa<0)qfa=i; if(c&&!g&&qfd<0)qfd=i; }
    qf=qfd>=0?qfd:qfa; if(qf<0){ fprintf(stderr,"no compute queue\n"); exit(1); }
    float prio=1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=qf; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    VkDevice dev; VKCHECK(vkCreateDevice(pd,&dci,nullptr,&dev));
    VkQueue queue; vkGetDeviceQueue(dev,qf,0,&queue);
    printf("device : %s  queue family=%d %s  M=%u N=%d barrier=%d\n",
           props.deviceName, qf, (qf==qfd)?"(dedicated compute/ACE)":"(universal)", M, N, barrier);

    VkDeviceSize bufsz=(VkDeviceSize)M*sizeof(float);
    Buf X=make_buf(pd,dev,bufsz), W=make_buf(pd,dev,bufsz), Y=make_buf(pd,dev,bufsz);
    auto initX=[&](){ float* x=(float*)X.map; for(uint32_t i=0;i<M;++i) x[i]=(i%7)*0.01f+0.1f; };
    auto initY=[&](){ float* yy=(float*)Y.map; for(uint32_t i=0;i<M;++i) yy[i]=(i%9)*0.01f+0.05f; };
    { float* w=(float*)W.map; for(uint32_t i=0;i<M;++i) w[i]=1.0f+((i%5)*0.001f); }
    initX(); initY();

    // descriptors: binding0=X, binding1=W
    VkDescriptorSetLayoutBinding b[2]={};
    for(int i=0;i<2;i++){ b[i].binding=i; b[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount=1; b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount=2; dslci.pBindings=b;
    VkDescriptorSetLayout dsl; VKCHECK(vkCreateDescriptorSetLayout(dev,&dslci,nullptr,&dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; VKCHECK(vkCreatePipelineLayout(dev,&plci,nullptr,&pl));

    auto mk_pipe=[&](const char* spv){
        auto code=read_spv(spv);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize=code.size()*4; smci.pCode=code.data();
        VkShaderModule sm; VKCHECK(vkCreateShaderModule(dev,&smci,nullptr,&sm));
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module=sm; cpci.stage.pName="main";
        cpci.layout=pl;
        VkPipeline p; VKCHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,nullptr,&p));
        return p;
    };
    VkPipeline pAdd=mk_pipe("layer_add.comp.spv"), pScale=mk_pipe("layer_scale.comp.spv");
    VkPipeline pRev=(rev||mix)?mk_pipe("layer_revadd.comp.spv"):VK_NULL_HANDLE;
    VkPipeline pCopy=(mix||serial)?mk_pipe("layer_copy.comp.spv"):VK_NULL_HANDLE;
    VkPipeline pBlend=serial?mk_pipe("layer_blend.comp.spv"):VK_NULL_HANDLE;

    // Pool: elem dset + dsetXY + dsetYX/setYW + (5*LAYERS) per-dispatch re-record sets.
    uint32_t recN = (rerec && mix) ? (uint32_t)5*LAYERS : 0;
    uint32_t maxSets = recN + 8;
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,maxSets*2};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=maxSets; dpci.poolSizeCount=1; dpci.pPoolSizes=&dps;
    VkDescriptorPool dpool; VKCHECK(vkCreateDescriptorPool(dev,&dpci,nullptr,&dpool));
    auto alloc_set=[&](){ VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool=dpool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
        VkDescriptorSet s; VKCHECK(vkAllocateDescriptorSets(dev,&dsai,&s)); return s; };
    auto write_set=[&](VkDescriptorSet s, VkBuffer b0, VkBuffer b1){
        VkDescriptorBufferInfo d0{b0,0,bufsz}, d1{b1,0,bufsz};
        VkWriteDescriptorSet wds[2]={{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        wds[0].dstSet=s; wds[0].dstBinding=0; wds[0].descriptorCount=1; wds[0].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[0].pBufferInfo=&d0;
        wds[1].dstSet=s; wds[1].dstBinding=1; wds[1].descriptorCount=1; wds[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[1].pBufferInfo=&d1;
        vkUpdateDescriptorSets(dev,2,wds,0,nullptr); };
    VkDescriptorSet dset=alloc_set();   write_set(dset, X.buf, W.buf);   // elem: b0=X, b1=W
    VkDescriptorSet dsetXY=VK_NULL_HANDLE, dsetYX=VK_NULL_HANDLE, setYW=VK_NULL_HANDLE;
    if(rev){
        dsetXY=alloc_set(); write_set(dsetXY, X.buf, Y.buf);  // k_revadd(x,y): dst=X, src=Y
        dsetYX=alloc_set(); write_set(dsetYX, Y.buf, X.buf);  // k_revadd(y,x): dst=Y, src=X
    }
    if(mix||serial){
        dsetXY=alloc_set(); write_set(dsetXY, X.buf, Y.buf);  // revadd/blend(x,y): b0=X, b1=Y
        dsetYX=alloc_set(); write_set(dsetYX, Y.buf, X.buf);  // copy(y,x):         b0=Y, b1=X
        setYW=alloc_set();  write_set(setYW,  Y.buf, W.buf);  // scale(y,w) & add(y,w): b0=Y, b1=W
    }
    // Per-dispatch descriptor sets for VK_RERECORD: one distinct set per dispatch in
    // the whole token (5*LAYERS), re-written every iteration (mirrors llama.cpp's
    // idx++ + updateDescriptorSets per token).
    std::vector<VkDescriptorSet> recset(recN);
    for(uint32_t k=0;k<recN;k++) recset[k]=alloc_set();

    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci2.queueFamilyIndex=qf;
    cpci2.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;  // allow per-iter re-record (VK_RERECORD)
    VkCommandPool cpool; VKCHECK(vkCreateCommandPool(dev,&cpci2,nullptr,&cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cb; VKCHECK(vkAllocateCommandBuffers(dev,&cbai,&cb));

    uint32_t groups=(M+255)/256;
    int   do_ts  = getenv("VK_TS")   != nullptr;   // per-dispatch GPU timestamps (decompose kernel vs gap)
    int   same_pl= getenv("VK_SAME") != nullptr;   // bind ONE pipeline (isolate per-dispatch rebind cost)
    double ts_ns = props.limits.timestampPeriod;
    uint32_t ndisp=(uint32_t)N*kpi;
    uint32_t nq = do_ts ? (2*ndisp) : 0;
    VkQueryPool qpool=VK_NULL_HANDLE;
    if(do_ts){
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType=VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount=nq;
        VKCHECK(vkCreateQueryPool(dev,&qpci,nullptr,&qpool));
    }
    if(same_pl) printf("mode   : VK_SAME (one pipeline bound once, no per-dispatch rebind; checksum will differ)\n");

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    auto emit_barrier=[&](){ if(barrier) vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr); };

    // llama.cpp-style global memory barrier: COMPUTE|TRANSFER stages, shader+transfer
    // read/write on both sides (matches ggml_vk_sync_buffers).
    VkMemoryBarrier mbL{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mbL.srcAccessMask=mbL.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|
                                        VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags stgL=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;
    // VK_RERECORD (mix): record ONE iteration into c, re-writing a fresh descriptor
    // set + push constants per dispatch and emitting the llama.cpp barrier, exactly
    // like ggml-vulkan re-records every token. Includes the CPU recording cost that
    // the pre-recorded path hides.
    auto record_mix_iter=[&](VkCommandBuffer c){
        VkBuffer b0[5]={Y.buf,Y.buf,X.buf,Y.buf,X.buf};   // copy(y,x) scale(y,w) revadd(x,y) add(y,w) revadd(x,y)
        VkBuffer b1[5]={X.buf,W.buf,Y.buf,W.buf,Y.buf};
        VkPipeline pp[5]={pCopy,pScale,pRev,pAdd,pRev};
        uint32_t gg[5]={groups,groups,groups,groups,groups};
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKCHECK(vkBeginCommandBuffer(c,&bi));
        uint32_t idx=0;
        for(int l=0;l<LAYERS;++l) for(int k=0;k<5;k++){
            write_set(recset[idx], b0[k], b1[k]);   // vkUpdateDescriptorSets every dispatch
            vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(uint32_t),&M);
            vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pp[k]);
            vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&recset[idx],0,nullptr);
            vkCmdDispatch(c,gg[k],1,1);
            if(barrier) vkCmdPipelineBarrier(c,stgL,stgL,0,1,&mbL,0,nullptr,0,nullptr);
            idx++;
        }
        VKCHECK(vkEndCommandBuffer(c));
    };
    bool rr = rerec && mix;

  if(!rr){
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(cb,&cbbi));
    if(do_ts) vkCmdResetQueryPool(cb,qpool,0,nq);
    vkCmdPushConstants(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(uint32_t),&M);
    uint32_t d=0;
    // one dispatch: optionally (re)bind descriptor set, bind pipeline (unless
    // VK_SAME), dispatch, barrier.
    auto oneg=[&](VkPipeline p, VkDescriptorSet s, uint32_t g){
        if(s!=VK_NULL_HANDLE) vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&s,0,nullptr);
        if(!same_pl) vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,p);
        if(do_ts) vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,qpool,2*d);
        vkCmdDispatch(cb,g,1,1);
        if(do_ts) vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,qpool,2*d+1);
        d++; emit_barrier();
    };
    auto one=[&](VkPipeline p, VkDescriptorSet s){ oneg(p,s,groups); };
    if(rev){
        // KLEN-kernel reverse-read chain, ping-pong via dsetYX/dsetXY (b0 is the
        // written buffer). Matches PM4/HIP: even k writes Y, odd k writes X.
        if(same_pl) vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pRev);
        for(int it=0;it<N;++it)
            for(int k=0;k<KLEN;++k) one(pRev, (k&1)?dsetXY:dsetYX);
    } else if(mix||serial){
        // Decode-like: 5 DISTINCT pipelines rebound every dispatch, repeated LAYERS
        // times (~300 dispatches/token). Mirrors PM4/HIP exactly. The middle blend
        // pipeline is the only difference: mix -> pRev (reverse-read stress),
        // serial -> pBlend (forward read, realistic decode chain).
        VkPipeline pMid = serial ? pBlend : pRev;
        for(int it=0;it<N;++it) for(int l=0;l<LAYERS;++l){
            one (pCopy, dsetYX);          // y = x
            one (pScale,setYW);           // y *= w
            one (pMid,  dsetXY);          // x = 0.5x + 0.5*(rev(y) | y)
            one (pAdd,  setYW);           // y += w
            one (pMid,  dsetXY);          // x = 0.5x + 0.5*(rev(y) | y)
        }
    } else {
        vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&dset,0,nullptr);
        if(same_pl) vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pAdd);
        for(int it=0;it<N;++it){ one(pAdd,VK_NULL_HANDLE); one(same_pl?pAdd:pScale,VK_NULL_HANDLE); one(pAdd,VK_NULL_HANDLE); }
    }
    VKCHECK(vkEndCommandBuffer(cb));
  } // end if(!rr)

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; VKCHECK(vkCreateFence(dev,&fci,nullptr,&fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;

    double t0,t1;
    if(rr){
        printf("mode   : VK_RERECORD (cmd buffer re-recorded + descriptors re-written every iteration, like llama.cpp decode)\n");
        // warmup: one re-recorded iteration
        record_mix_iter(cb);
        VKCHECK(vkQueueSubmit(queue,1,&si,fence));
        VKCHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));
        VKCHECK(vkResetFences(dev,1,&fence));
        initX();
        t0=now_s();
        for(int it=0;it<N;++it){
            record_mix_iter(cb);                                  // CPU: begin + 5x(update/push/bind/dispatch/barrier) + end
            VKCHECK(vkQueueSubmit(queue,1,&si,fence));
            VKCHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));
            VKCHECK(vkResetFences(dev,1,&fence));
        }
        t1=now_s();
    } else {
        // warmup (also ramps clocks / builds pipelines), then re-init x for the timed run
        VKCHECK(vkQueueSubmit(queue,1,&si,fence));
        VKCHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));
        VKCHECK(vkResetFences(dev,1,&fence));
        initX(); if(rev) initY();

        t0=now_s();
        VKCHECK(vkQueueSubmit(queue,1,&si,fence));
        VKCHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));
        t1=now_s();
    }
    double us=(t1-t0)*1e6;

    float* x=(float*)X.map; double sum=0; for(uint32_t i=0;i<M;++i) sum+=x[i];
    printf("result : checksum(x)=%.6f x[0]=%.6f x[%u]=%.6f\n", sum, x[0], M-1, x[M-1]);
    printf("         CPU wall %.2f us  PER-ITER(%d kern)=%.3f us  PER-DISPATCH=%.3f us\n",
           us, kpi, us/N, us/(double)(N*kpi));

    if(do_ts && !rr){
        std::vector<uint64_t> ts(nq);
        VKCHECK(vkGetQueryPoolResults(dev,qpool,0,nq,nq*sizeof(uint64_t),ts.data(),
                sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
        auto us_of=[&](uint64_t a,uint64_t b){ return (double)(b-a)*ts_ns/1000.0; };
        // skip first/last 30 dispatches (clock ramp / tail)
        uint32_t skip=30; if(skip*2>=ndisp) skip=0;
        double kern_sum=0, per_sum=0; uint32_t kn=0, pn=0;
        for(uint32_t i=skip;i<ndisp-skip;++i){
            kern_sum += us_of(ts[2*i], ts[2*i+1]); kn++;
            if(i+1<ndisp){ per_sum += us_of(ts[2*i+1], ts[2*(i+1)+1]); pn++; }
        }
        double gpu_span = us_of(ts[0], ts[2*(ndisp-1)+1]);
        printf("  [GPU timestamps] whole-pipeline GPU span = %.2f us  (CPU wall = %.2f us; ratio %.3f)\n",
               gpu_span, us, us/gpu_span);
        printf("  [GPU timestamps] per-dispatch: kernel(TOP->BOT) mean %.3f us | period(BOT->BOT) mean %.3f us\n",
               kn?kern_sum/kn:0, pn?per_sum/pn:0);
        printf("  [GPU timestamps] => of the %.3f us/dispatch, ~%.3f us is kernel exec, ~%.3f us is gap(barrier+CP+drain)\n",
               pn?per_sum/pn:0, kn?kern_sum/kn:0, (pn&&kn)?(per_sum/pn - kern_sum/kn):0);
        vkDestroyQueryPool(dev,qpool,nullptr);
    }
    return 0;
}
