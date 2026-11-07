// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <utility>
#include <vector>

#include "qt_common/qt_common.h"

#include "common/dynamic_library.h"
#include "common/logging.h"
#include "qt_common/util/vk.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "vulkan/vulkan_core.h"

class QWindow;

namespace VkDeviceInfo {
Record::Record(std::string_view name_, const std::vector<VkPresentModeKHR>& vsync_modes_,
               bool has_broken_compute_, u64 device_memory_, u64 memory_budget_, bool integrated_)
    : name{name_}, vsync_support{vsync_modes_}, has_broken_compute{has_broken_compute_},
      device_memory{device_memory_}, memory_budget{memory_budget_}, integrated{integrated_} {}

Record::~Record() = default;

void PopulateRecords(std::vector<Record>& records, QWindow* window) try {
    using namespace Vulkan;

    // Create a test window with a Vulkan surface type for checking present modes.
    QWindow test_window(window);
    test_window.setSurfaceType(QWindow::VulkanSurface);
    test_window.create();
    auto wsi = QtCommon::GetWindowSystemInfo(&test_window);

    vk::InstanceDispatch dld;
    const auto library = OpenLibrary();
    const vk::Instance instance = CreateInstance(*library, dld, VK_API_VERSION_1_1, wsi.type);
    const std::vector<VkPhysicalDevice> physical_devices = instance.EnumeratePhysicalDevices();
    vk::SurfaceKHR surface = CreateSurface(instance, wsi);

    records.clear();
    records.reserve(physical_devices.size());
    for (const VkPhysicalDevice device : physical_devices) {
        const auto physical_device = vk::PhysicalDevice(device, dld);
        std::string name = physical_device.GetProperties().deviceName;

        const std::vector<VkPresentModeKHR> present_modes =
            physical_device.GetSurfacePresentModesKHR(*surface);

        VkPhysicalDeviceDriverProperties driver_properties{};
        driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        driver_properties.pNext = nullptr;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        properties.pNext = &driver_properties;
        dld.vkGetPhysicalDeviceProperties2(physical_device, &properties);

        const auto driverID = driver_properties.driverID;

        bool has_broken_compute{
            Vulkan::Device::CheckBrokenCompute(driverID, properties.properties.driverVersion)};

        std::string driver_string = Vulkan::vk::GetDriverName(driver_properties);

        if (driver_string.empty())
            driver_string = "Unknown";

        name = fmt::format("{} ({})", name, driver_string);

        // Device memory for the Texture Cache Memory Limit's Auto value: the device-local heaps,
        // and the largest one's budget (what the texture cache follows in game).
        bool has_memory_budget = false;
        for (const VkExtensionProperties& extension :
             physical_device.EnumerateDeviceExtensionProperties()) {
            has_memory_budget |=
                std::string_view{extension.extensionName} == VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
        }
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        const VkPhysicalDeviceMemoryProperties memory =
            physical_device.GetMemoryProperties(has_memory_budget ? &budget : nullptr)
                .memoryProperties;
        u64 device_memory = 0;
        u64 main_heap_size = 0;
        u64 memory_budget = 0;
        for (u32 heap = 0; heap < memory.memoryHeapCount; ++heap) {
            if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
                continue;
            }
            device_memory += memory.memoryHeaps[heap].size;
            if (memory.memoryHeaps[heap].size > main_heap_size) {
                main_heap_size = memory.memoryHeaps[heap].size;
                memory_budget = has_memory_budget ? budget.heapBudget[heap] : main_heap_size;
            }
        }
        const bool integrated =
            properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

        records.push_back(VkDeviceInfo::Record(name, present_modes, has_broken_compute,
                                               device_memory, memory_budget, integrated));
    }
} catch (const Vulkan::vk::Exception& exception) {
    LOG_ERROR(Frontend, "Failed to enumerate devices with error: {}", exception.what());
}

} // namespace VkDeviceInfo
