#ifndef OPENMM_INTERNAL_COLVARS_KERNELS_H
#define OPENMM_INTERNAL_COLVARS_KERNELS_H

#include "openmm/KernelImpl.h"

#include <string>

namespace OpenMM {

class ColvarsForceImpl;
class ContextImpl;
class Platform;

/**
 * Sets up a ColvarsForce to run after every other force of a GPU context, where the
 * completed force buffer makes total-force sampling (ABF/eABF) possible. Platforms
 * without this kernel use the CustomCPPForce path.
 */
class InitializeDeferredColvarsKernel : public KernelImpl {
public:
    static std::string Name() {
        return "InitializeDeferredColvars";
    }

    InitializeDeferredColvarsKernel(std::string name, const Platform& platform)
        : KernelImpl(name, platform) {
    }

    virtual void initialize(ContextImpl& context, ColvarsForceImpl& force) = 0;
};

/**
 * Registers the OpenMM-Colvars kernel factories on every platform that does not have
 * them yet. The CUDA plugin, when installed, registers its factories when OpenMM loads
 * it, so they take precedence over the fallbacks registered here.
 */
void registerColvarsKernelFactories();

/**
 * Registers the CUDA kernel factories. Defined only in builds with CUDA kernels, either
 * in the core library or in the CUDA plugin.
 */
void registerCudaColvarsKernelFactories(Platform& platform);

} // namespace OpenMM

#endif
