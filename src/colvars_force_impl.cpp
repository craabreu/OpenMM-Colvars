#include "openmm/internal/ColvarsForceImpl.h"
#include "openmm/internal/ColvarsKernels.h"
#include "openmm/internal/OpenMMColvarProxy.h"

#include "openmm/CustomExternalForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/ContextImpl.h"

#include <algorithm>
#include <cstdint>

using namespace OpenMM;

ColvarsForceImpl::ColvarsForceImpl(const ColvarsForce& owner)
    : CustomCPPForceImpl(owner),
      owner_(owner),
      usesDeferredCudaComputation_(false)
{
}

namespace {

bool isOwnerLastForce(const System& system, const Force& owner)
{
    const int numForces = system.getNumForces();
    return numForces > 0 && &system.getForce(numForces - 1) == &owner;
}

bool isSafeAfterColvarsForTotalForces(const Force& force)
{
    const auto* external = dynamic_cast<const CustomExternalForce*>(&force);
    return external != nullptr && external->getEnergyFunction() == "0";
}

bool hasOnlySafeForcesAfterOwner(const System& system, const Force& owner)
{
    bool afterOwner = false;
    for (int forceIndex = 0; forceIndex < system.getNumForces(); ++forceIndex) {
        const Force& force = system.getForce(forceIndex);
        if (&force == &owner) {
            afterOwner = true;
            continue;
        }
        if (afterOwner && !isSafeAfterColvarsForTotalForces(force)) {
            return false;
        }
    }
    return afterOwner;
}

void validateTotalForceOrdering(const ContextImpl& context,
                                const ColvarsForce& owner,
                                const OpenMMColvarProxy& proxy)
{
    if (!proxy.total_forces_enabled() || isOwnerLastForce(context.getSystem(), owner) ||
        hasOnlySafeForcesAfterOwner(context.getSystem(), owner)) {
        return;
    }
    throw OpenMMException(
        "Colvars configurations that request total forces, such as ABF/eABF, "
        "require ColvarsForce to execute after every physical Force on Reference "
        "and CPU platforms so the current OpenMM force buffer is complete.");
}

void validateTotalForceConfiguration(ContextImpl& context,
                                     const ColvarsForce& owner,
                                     const OpenMMColvarProxy& proxy,
                                     bool usesDeferredCudaComputation)
{
    if (!proxy.total_forces_enabled()) {
        return;
    }

    int colvarsForces = 0;
    for (int forceIndex = 0; forceIndex < context.getSystem().getNumForces(); ++forceIndex) {
        if (dynamic_cast<const ColvarsForce*>(&context.getSystem().getForce(forceIndex)) != nullptr) {
            ++colvarsForces;
        }
    }
    if (colvarsForces > 1) {
        throw OpenMMException(
            "A System containing a Colvars configuration that requests total forces "
            "may contain only one ColvarsForce. Put multiple Colvars biases in one configuration.");
    }

    const std::string platformName = context.getPlatform().getName();
    if (usesDeferredCudaComputation) {
        return;
    }
    if (platformName != "Reference" && platformName != "CPU") {
        throw OpenMMException(
            "Colvars total-force sampling is unsupported on platform \"" + platformName +
            "\" because the completed force buffer is unavailable. Use Reference or CPU, "
            "or use CUDA with the CUDA kernels of OpenMM-Colvars (a build with "
            "OPENMM_COLVARS_ENABLE_CUDA_KERNELS=ON, or its CUDA plugin).");
    }
    validateTotalForceOrdering(context, owner, proxy);
}

void validateNonDeferredForceGroup(const ColvarsForce& owner)
{
    if (owner.getForceGroup() == 31) {
        throw OpenMMException(
            "ColvarsForce group 31 is unsupported on the non-deferred CustomCPPForce path "
            "because this OpenMM dependency uses a signed force-group shift. Use group 0-30, "
            "or use a deferred CUDA build.");
    }
}

} // namespace

const ColvarsForce& ColvarsForceImpl::getOwner() const
{
    return owner_;
}

void ColvarsForceImpl::initialize(ContextImpl& context)
{
    registerColvarsKernelFactories();
    Platform& platform = context.getPlatform();
    if (platform.supportsKernels({InitializeDeferredColvarsKernel::Name()})) {
        proxy_ = std::make_unique<OpenMMColvarProxy>(owner_, context.getSystem());
        validateTotalForceConfiguration(context, owner_, *proxy_, true);
        deferredKernel_ =
            platform.createKernel(InitializeDeferredColvarsKernel::Name(), context);
        deferredKernel_.getAs<InitializeDeferredColvarsKernel>().initialize(context, *this);
        usesDeferredCudaComputation_ = true;
        return;
    }
    validateNonDeferredForceGroup(owner_);
    CustomCPPForceImpl::initialize(context);
    proxy_ = std::make_unique<OpenMMColvarProxy>(owner_, context.getSystem());
    validateTotalForceConfiguration(context, owner_, *proxy_, false);
}

double ColvarsForceImpl::calcForcesAndEnergy(ContextImpl& context,
                                             bool includeForces,
                                             bool includeEnergy,
                                             int groups)
{
    if (usesDeferredCudaComputation_) {
        return 0.0;
    }
    return CustomCPPForceImpl::calcForcesAndEnergy(
        context, includeForces, includeEnergy, groups);
}

double ColvarsForceImpl::computeForce(ContextImpl& context,
                                      const std::vector<Vec3>& positions,
                                      std::vector<Vec3>& forces)
{
    std::fill(forces.begin(), forces.end(), Vec3());
    if (proxy_ == nullptr) {
        proxy_ = std::make_unique<OpenMMColvarProxy>(owner_, context.getSystem());
    }

    const double energy_kj_per_mol = proxy_->calculate(context, positions);
    proxy_->addForcesToOpenMM(forces);
    return energy_kj_per_mol;
}

std::map<std::string, double> ColvarsForceImpl::getDefaultParameters()
{
    std::map<std::string, double> parameters;
    if (owner_.getAlchemyEnabled()) {
        parameters[owner_.getLambdaParameterName()] = 0.0;
        parameters[owner_.getLambdaDerivativeParameterName()] = 0.0;
    }
    return parameters;
}
