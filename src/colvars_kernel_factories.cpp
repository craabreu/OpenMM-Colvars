#include "openmm/internal/AlchemicalPMEKernels.h"
#include "openmm/internal/ColvarsKernels.h"

#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"

#include <mutex>

using namespace OpenMM;

namespace {

#ifndef OPENMM_COLVARS_ENABLE_CUDA_KERNELS

const char* missingCudaKernelsMessage =
    "The CUDA AlchemicalPMEForce kernel is not available. Build OpenMM-Colvars with "
    "-DOPENMM_COLVARS_ENABLE_CUDA_KERNELS=ON and OpenMM CUDA internal headers, or "
    "install its CUDA plugin.";

class UnavailableCudaAlchemicalPMEForceKernel final : public CalcAlchemicalPMEForceKernel {
public:
    UnavailableCudaAlchemicalPMEForceKernel(std::string name, const Platform& platform)
        : CalcAlchemicalPMEForceKernel(name, platform) {
    }

    void initialize(ContextImpl& context, const AlchemicalPMEForce& force) override {
        throw OpenMMException(missingCudaKernelsMessage);
    }

    double execute(ContextImpl& context, bool includeForces, bool includeEnergy) override {
        throw OpenMMException(missingCudaKernelsMessage);
    }
};

class UnavailableCudaAlchemicalPMEKernelFactory final : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const override {
        if (name == CalcAlchemicalPMEForceKernel::Name()) {
            return new UnavailableCudaAlchemicalPMEForceKernel(name, platform);
        }
        throw OpenMMException("Tried to create an illegal CUDA AlchemicalPMEForce kernel");
    }
};

#endif

std::mutex registerFactoriesMutex;

} // namespace

void OpenMM::registerColvarsKernelFactories()
{
    std::lock_guard<std::mutex> lock(registerFactoriesMutex);
    for (int i = 0; i < Platform::getNumPlatforms(); ++i) {
        Platform& platform = Platform::getPlatform(i);
        if ((platform.getName() == "Reference" || platform.getName() == "CPU") &&
            !platform.supportsKernels({CalcAlchemicalPMEForceKernel::Name()})) {
            registerReferenceAlchemicalPMEKernelFactory(platform);
        }
        if (platform.getName() != "CUDA") {
            continue;
        }
#ifdef OPENMM_COLVARS_ENABLE_CUDA_KERNELS
        if (!platform.supportsKernels({CalcAlchemicalPMEForceKernel::Name(),
                                       InitializeDeferredColvarsKernel::Name()})) {
            registerCudaColvarsKernelFactories(platform);
        }
#else
        if (!platform.supportsKernels({CalcAlchemicalPMEForceKernel::Name()})) {
            platform.registerKernelFactory(CalcAlchemicalPMEForceKernel::Name(),
                                           new UnavailableCudaAlchemicalPMEKernelFactory());
        }
#endif
    }
}
