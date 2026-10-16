#include "openmm/internal/ColvarsKernels.h"

#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/internal/AlchemicalPMEKernels.h"
#include "openmm/internal/ColvarsForceImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/windowsExport.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace OpenMM;

namespace {

const char* deferredColvarsForceSource = R"(
KERNEL void addForces(GLOBAL const real* RESTRICT forces,
                      GLOBAL mm_long* RESTRICT forceBuffers,
                      GLOBAL int* RESTRICT atomIndex) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        int index = atomIndex[atom];
        forceBuffers[atom] += (mm_long) (forces[3*index]*0x100000000);
        forceBuffers[atom+PADDED_NUM_ATOMS] +=
            (mm_long) (forces[3*index+1]*0x100000000);
        forceBuffers[atom+2*PADDED_NUM_ATOMS] +=
            (mm_long) (forces[3*index+2]*0x100000000);
    }
}
)";

class DeferredCudaColvarsComputation final : public ComputeContext::ForcePostComputation {
public:
    DeferredCudaColvarsComputation(ContextImpl& context,
                                   CudaContext& computeContext,
                                   ColvarsForceImpl& force)
        : context_(context),
          computeContext_(computeContext),
          cudaContext_(computeContext),
          force_(force),
          forceGroupFlag_(std::uint32_t{1} <<
                          static_cast<unsigned>(force.getOwner().getForceGroup()))
    {
        ContextSelector selector(computeContext_);
        const int numParticles = context_.getSystem().getNumParticles();
        const int paddedNumParticles = computeContext_.getPaddedNumAtoms();
        positions_.resize(numParticles);
        forces_.resize(numParticles);
        floatForces_.resize(3 * numParticles);
        const int elementSize =
            computeContext_.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
        forceArray_.initialize(computeContext_,
                               3 * numParticles,
                               elementSize,
                               "deferredColvarsForces");
        std::map<std::string, std::string> defines;
        defines["NUM_ATOMS"] = computeContext_.intToString(numParticles);
        defines["PADDED_NUM_ATOMS"] =
            computeContext_.intToString(computeContext_.getPaddedNumAtoms());
        ComputeProgram program =
            computeContext_.compileProgram(deferredColvarsForceSource, defines);
        addForcesKernel_ = program->createKernel("addForces");
        addForcesKernel_->addArg(forceArray_);
        addForcesKernel_->addArg(computeContext_.getLongForceBuffer());
        addForcesKernel_->addArg(computeContext_.getAtomIndexArray());
    }

    double computeForceAndEnergy(bool includeForces,
                                 bool includeEnergy,
                                 int groups) override
    {
        (void) includeEnergy;
        if ((static_cast<std::uint32_t>(groups) & forceGroupFlag_) == 0) {
            return 0.0;
        }

        cudaContext_.flushQueue();
        context_.getPositions(positions_);
        const double energy = force_.computeForce(context_, positions_, forces_);
        if (!includeForces) {
            return energy;
        }

        ContextSelector selector(computeContext_);
        if (computeContext_.getUseDoublePrecision()) {
            forceArray_.upload(reinterpret_cast<const double*>(forces_.data()));
        } else {
            for (int i = 0; i < static_cast<int>(forces_.size()); ++i) {
                floatForces_[3 * i] = static_cast<float>(forces_[i][0]);
                floatForces_[3 * i + 1] = static_cast<float>(forces_[i][1]);
                floatForces_[3 * i + 2] = static_cast<float>(forces_[i][2]);
            }
            forceArray_.upload(floatForces_);
        }
        addForcesKernel_->execute(computeContext_.getNumAtoms());
        return energy;
    }

private:
    ContextImpl& context_;
    ComputeContext& computeContext_;
    CudaContext& cudaContext_;
    ColvarsForceImpl& force_;
    std::uint32_t forceGroupFlag_;
    ComputeArray forceArray_;
    ComputeKernel addForcesKernel_;
    std::vector<Vec3> positions_;
    std::vector<Vec3> forces_;
    std::vector<float> floatForces_;
};

class KeepDeferredColvarsLast final : public ComputeContext::ForcePreComputation {
public:
    KeepDeferredColvarsLast(ComputeContext& context,
                            ComputeContext::ForcePostComputation* colvars)
        : context_(context), colvars_(colvars)
    {
    }

    void computeForceAndEnergy(bool includeForces,
                               bool includeEnergy,
                               int groups) override
    {
        (void) includeForces;
        (void) includeEnergy;
        (void) groups;
        auto& computations = context_.getPostComputations();
        const auto found = std::find(computations.begin(), computations.end(), colvars_);
        if (found != computations.end() && found + 1 != computations.end()) {
            computations.erase(found);
            computations.push_back(colvars_);
        }
    }

private:
    ComputeContext& context_;
    ComputeContext::ForcePostComputation* colvars_;
};

} // namespace

namespace {

class CudaInitializeDeferredColvarsKernel final : public InitializeDeferredColvarsKernel {
public:
    CudaInitializeDeferredColvarsKernel(std::string name, const Platform& platform)
        : InitializeDeferredColvarsKernel(name, platform) {
    }

    void initialize(ContextImpl& context, ColvarsForceImpl& force) override {
        auto* data =
            reinterpret_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
        if (data == nullptr || data->contexts.size() != 1) {
            throw OpenMMException(
                "CUDA ColvarsForce currently supports a single CUDA context only");
        }
        CudaContext& computeContext = *data->contexts[0];
        auto* deferred = new DeferredCudaColvarsComputation(context, computeContext, force);
        computeContext.addPostComputation(deferred);
        computeContext.addPreComputation(new KeepDeferredColvarsLast(computeContext, deferred));
    }
};

class CudaColvarsKernelFactory final : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const override {
        if (name == InitializeDeferredColvarsKernel::Name()) {
            return new CudaInitializeDeferredColvarsKernel(name, platform);
        }
        throw OpenMMException("Tried to create an illegal CUDA ColvarsForce kernel");
    }
};

} // namespace

void OpenMM::registerCudaColvarsKernelFactories(Platform& platform)
{
    registerCudaAlchemicalPMEKernelFactory(platform);
    platform.registerKernelFactory(InitializeDeferredColvarsKernel::Name(),
                                   new CudaColvarsKernelFactory());
}

#ifdef OPENMM_COLVARS_BUILDING_CUDA_PLUGIN

// Entry points of the CUDA plugin, which OpenMM calls when it loads its plugins

extern "C" OPENMM_EXPORT void registerPlatforms()
{
}

extern "C" OPENMM_EXPORT void registerKernelFactories()
{
    for (int i = 0; i < Platform::getNumPlatforms(); ++i) {
        Platform& platform = Platform::getPlatform(i);
        if (platform.getName() == "CUDA") {
            registerCudaColvarsKernelFactories(platform);
        }
    }
}

#endif
