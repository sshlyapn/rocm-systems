// Vulkan compute inter-kernel gap microbenchmark.
//
// Records K compute dispatches into a single command buffer on one compute
// queue, optionally separated by a SHADER_WRITE->SHADER_READ pipeline barrier
// (the same kind ggml-vulkan emits between dependent ops). Measures, on the GPU
// timeline, the duration of each dispatch and the gap between consecutive
// dispatches using TOP_OF_PIPE / BOTTOM_OF_PIPE timestamps.
//
//   start[k] : TOP_OF_PIPE timestamp written just before dispatch k
//   end[k]   : BOTTOM_OF_PIPE timestamp written just after dispatch k
//   kernel[k] = end[k]   - start[k]      (pure dispatch GPU time)
//   gap[k]    = start[k+1] - end[k]      (barrier + CP prelude = inter-kernel gap)
//
// Usage: vk_gap_test [K] [spin] [n] [barrier 0/1] [perdispatch 0/1]
//
// Build: see build.sh (needs gap.comp.spv next to the binary or in CWD).

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define VKCHECK(x)                                                                       \
    do {                                                                                 \
        VkResult _r = (x);                                                               \
        if (_r != VK_SUCCESS) {                                                          \
            fprintf(stderr, "VK error %d at %s:%d\n", (int)_r, __FILE__, __LINE__);       \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

static std::vector<uint32_t> read_spv(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open SPIR-V %s\n", path);
        exit(1);
    }
    size_t sz = (size_t)f.tellg();
    std::vector<uint32_t> data(sz / 4);
    f.seekg(0);
    f.read((char *)data.data(), sz);
    return data;
}

static uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    fprintf(stderr, "no suitable memory type\n");
    exit(1);
}

int main(int argc, char ** argv) {
    int      K       = argc > 1 ? atoi(argv[1]) : 2000;
    uint32_t spin    = argc > 2 ? (uint32_t)atoi(argv[2]) : 0;
    uint32_t n       = argc > 3 ? (uint32_t)atoi(argv[3]) : 4096;
    int      barrier = argc > 4 ? atoi(argv[4]) : 1;
    int      perdisp = argc > 5 ? atoi(argv[5]) : 1;

    // ---- instance ----
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    VkInstance instance;
    VKCHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(instance, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(instance, &ndev, devs.data());
    if (ndev == 0) { fprintf(stderr, "no Vulkan devices\n"); exit(1); }
    VkPhysicalDevice pd = devs[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);

    // ---- pick a compute-capable queue family; prefer a dedicated (non-graphics) one ----
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfs.data());
    int qf = -1, qf_dedicated = -1, qf_any = -1;
    for (uint32_t i = 0; i < nqf; i++) {
        bool comp = qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT;
        bool gfx  = qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
        if (comp && qf_any < 0) qf_any = (int)i;
        if (comp && !gfx && qf_dedicated < 0) qf_dedicated = (int)i;
    }
    qf = qf_dedicated >= 0 ? qf_dedicated : qf_any;
    if (qf < 0) { fprintf(stderr, "no compute queue\n"); exit(1); }
    bool dedicated = (qf == qf_dedicated);
    uint32_t tsbits = qfs[qf].timestampValidBits;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = (uint32_t)qf;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    VkDevice dev;
    VKCHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, (uint32_t)qf, 0, &queue);

    double ts_period_ns = props.limits.timestampPeriod;

    printf("device : %s  (apiVersion %u.%u.%u)\n", props.deviceName,
           VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));
    printf("queue  : family=%d %s  timestampValidBits=%u  timestampPeriod=%.4f ns\n",
           qf, dedicated ? "(dedicated compute / ACE)" : "(universal gfx+compute)", tsbits, ts_period_ns);
    printf("config : K=%d dispatches  spin=%u  n=%u (%u groups)  barrier=%d  perdispatch=%d\n",
           K, spin, n, (n + 255) / 256, barrier, perdisp);

    // ---- storage buffer (device local) ----
    VkDeviceSize bufsz = (VkDeviceSize)n * sizeof(float);
    if (bufsz < 256) bufsz = 256;
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = bufsz;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf;
    VKCHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = find_mem_type(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory mem;
    VKCHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
    VKCHECK(vkBindBufferMemory(dev, buf, mem, 0));

    // ---- descriptor / pipeline ----
    VkDescriptorSetLayoutBinding b0{};
    b0.binding         = 0;
    b0.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b0.descriptorCount = 1;
    b0.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 1;
    dslci.pBindings    = &b0;
    VkDescriptorSetLayout dsl;
    VKCHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    VkPipelineLayout pl;
    VKCHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    auto spv = read_spv("gap.comp.spv");
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv.size() * 4;
    smci.pCode    = spv.data();
    VkShaderModule sm;
    VKCHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName  = "main";
    cpci.layout       = pl;
    VkPipeline pipe;
    VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    VkDescriptorPoolSize dps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &dps;
    VkDescriptorPool dpool;
    VKCHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool     = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dsl;
    VkDescriptorSet dset;
    VKCHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
    VkDescriptorBufferInfo dbi{ buf, 0, bufsz };
    VkWriteDescriptorSet wds{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    wds.dstSet          = dset;
    wds.dstBinding      = 0;
    wds.descriptorCount = 1;
    wds.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo     = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

    // ---- query pool (timestamps) ----
    uint32_t nq = perdisp ? (uint32_t)(2 * K) : 2;
    VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = nq;
    VkQueryPool qpool;
    VKCHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qpool));

    // ---- command pool / buffer ----
    VkCommandPoolCreateInfo cpci2{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci2.queueFamilyIndex = (uint32_t)qf;
    VkCommandPool cpool;
    VKCHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool        = cpool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VKCHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    uint32_t pcvals[2] = { spin, n };
    uint32_t groups    = (n + 255) / 256;

    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkCommandBufferBeginInfo cbbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VKCHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdResetQueryPool(cb, qpool, 0, nq);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcvals), pcvals);

    if (perdisp) {
        for (int k = 0; k < K; k++) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, (uint32_t)(2 * k));
            vkCmdDispatch(cb, groups, 1, 1);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, (uint32_t)(2 * k + 1));
            if (barrier && k < K - 1) {
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            }
        }
    } else {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
        for (int k = 0; k < K; k++) {
            vkCmdDispatch(cb, groups, 1, 1);
            if (barrier && k < K - 1) {
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            }
        }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
    }
    VKCHECK(vkEndCommandBuffer(cb));

    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VKCHECK(vkCreateFence(dev, &fci, nullptr, &fence));

    // warmup submit (build caches, ramp clocks) - not timed
    {
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        VKCHECK(vkQueueSubmit(queue, 1, &si, fence));
        VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VKCHECK(vkResetFences(dev, 1, &fence));
    }

    // timed submit
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    VKCHECK(vkQueueSubmit(queue, 1, &si, fence));
    VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

    std::vector<uint64_t> ts(nq);
    VKCHECK(vkGetQueryPoolResults(dev, qpool, 0, nq, nq * sizeof(uint64_t), ts.data(),
                                  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

    auto to_us  = [&](uint64_t ticks) { return (double)ticks * ts_period_ns / 1000.0; };
    auto to_usi = [&](int64_t ticks) { return (double)ticks * ts_period_ns / 1000.0; };

    if (perdisp) {
        std::vector<double> kern, gap, period;
        for (int k = 0; k < K; k++) {
            // kernel: TOP(k) -> BOTTOM(k). Note TOP is a CP-immediate write so this
            // can include CP-run-ahead; treat as approximate.
            kern.push_back(to_usi((int64_t)(ts[2 * k + 1] - ts[2 * k])));
            if (k < K - 1) {
                // gap: BOTTOM(k) -> TOP(k+1). Signed: negative means the CP set up
                // dispatch k+1 before dispatch k finished (overlap, gap hidden).
                gap.push_back(to_usi((int64_t)(ts[2 * (k + 1)] - ts[2 * k + 1])));
                // period: BOTTOM(k) -> BOTTOM(k+1). Robust (both EOP). = kernel+gap.
                period.push_back(to_usi((int64_t)(ts[2 * (k + 1) + 1] - ts[2 * k + 1])));
            }
        }
        // drop first 50 dispatches (clock ramp / cold) from stats
        int skip = std::min(50, K / 10);
        auto stats = [&](std::vector<double> v) {
            std::vector<double> s(v.begin() + std::min((size_t)skip, v.size()), v.end());
            std::sort(s.begin(), s.end());
            double sum = 0;
            for (double x : s) sum += x;
            double mean = s.empty() ? 0 : sum / s.size();
            double med  = s.empty() ? 0 : s[s.size() / 2];
            double p5   = s.empty() ? 0 : s[(size_t)(s.size() * 0.05)];
            double p95  = s.empty() ? 0 : s[(size_t)(s.size() * 0.95)];
            return std::vector<double>{ mean, med, p5, p95, s.empty() ? 0 : s.front(), s.empty() ? 0 : s.back() };
        };
        auto ks = stats(kern);
        auto ps = stats(period);
        printf("\n  steady-state (skipped first %d dispatches):\n", skip);
        printf("  kernel TOP->BOT : mean %8.3f  median %8.3f  p5 %8.3f  p95 %8.3f us\n",
               ks[0], ks[1], ks[2], ks[3]);
        printf("  period BOT->BOT : mean %8.3f  median %8.3f  p5 %8.3f  p95 %8.3f us  (robust per-dispatch)\n",
               ps[0], ps[1], ps[2], ps[3]);
        auto gs = stats(gap);
        printf("  gap   BOT->TOP  : mean %8.3f  median %8.3f  p5 %8.3f  p95 %8.3f us  (signed; <0 = CP overlap)\n",
               gs[0], gs[1], gs[2], gs[3]);
        double total = to_usi((int64_t)(ts[2 * (K - 1) + 1] - ts[0]));
        printf("  total span = %.2f us over %d dispatches = %.3f us/dispatch  (barrier=%d)\n",
               total, K, total / K, barrier);
    } else {
        double total = to_us(ts[1] - ts[0]);
        printf("\n  total span = %.2f us over %d dispatches = %.4f us/dispatch  (barrier=%d)\n",
               total, K, total / K, barrier);
    }

    vkDestroyFence(dev, fence, nullptr);
    vkDestroyQueryPool(dev, qpool, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDescriptorPool(dev, dpool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyBuffer(dev, buf, nullptr);
    vkFreeMemory(dev, mem, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
