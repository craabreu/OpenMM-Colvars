#include "openmm/ColvarsCollectiveVariable.h"
#include "openmm/internal/ColvarsCollectiveVariableImpl.h"
#include "openmm/internal/OpenMMColvarProxy.h"

#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/serialization/SerializationProxy.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <typeinfo>

using namespace OpenMM;

namespace {

void validate_nonempty(const std::string& value, const std::string& name)
{
    if (value.empty()) {
        throw OpenMMException("ColvarsCollectiveVariable: " + name + " must not be empty");
    }
}

std::string to_lower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

// The colvar receives its generalized force through the scripted-forces hook of
// Colvars, which the proxy implements. Colvars rejects a repeated keyword, so the
// keyword is added only if the configuration does not have it.
std::string full_config(const std::string& configString)
{
    if (to_lower(configString).find("scriptedcolvarforces") != std::string::npos) {
        return configString;
    }
    return configString + "\nscriptedColvarForces on\n";
}

ColvarsForce make_settings(const ColvarsCollectiveVariable& owner)
{
    // The output prefix is required by ColvarsForce but unused: the proxy writes no
    // files when it reports a colvar
    ColvarsForce settings = ColvarsForce::fromConfigString(
        full_config(owner.getConfigString()), "colvars");
    settings.setAlchemyEnabled(false);
    settings.setPbcMode(owner.getPbcMode());
    return settings;
}

class ColvarsCollectiveVariableProxy final : public SerializationProxy {
public:
    ColvarsCollectiveVariableProxy() : SerializationProxy("ColvarsCollectiveVariable") {}

    void serialize(const void* object, SerializationNode& node) const override
    {
        const auto& force = *reinterpret_cast<const ColvarsCollectiveVariable*>(object);
        node.setIntProperty("version", 1);
        node.setIntProperty("forceGroup", force.getForceGroup());
        node.setStringProperty("name", force.getName());
        node.setStringProperty("configString", force.getConfigString());
        node.setStringProperty("colvarName", force.getColvarName());
        node.setDoubleProperty("valueScale", force.getValueScale());
        node.setIntProperty("pbcMode", static_cast<int>(force.getPbcMode()));
    }

    void* deserialize(const SerializationNode& node) const override
    {
        if (node.getIntProperty("version") != 1) {
            throw OpenMMException("Unsupported version number");
        }
        auto* force = new ColvarsCollectiveVariable(node.getStringProperty("configString"),
                                                    node.getStringProperty("colvarName"),
                                                    node.getDoubleProperty("valueScale"));
        try {
            force->setForceGroup(node.getIntProperty("forceGroup", 0));
            force->setName(node.getStringProperty("name", force->getName()));
            force->setPbcMode(static_cast<ColvarsForce::PbcMode>(node.getIntProperty("pbcMode")));
        } catch (...) {
            delete force;
            throw;
        }
        return force;
    }
};

// Registers the serialization proxy whenever this class is linked in
[[maybe_unused]] const bool proxyRegistered = [] {
    SerializationProxy::registerProxy(typeid(ColvarsCollectiveVariable),
                                      new ColvarsCollectiveVariableProxy());
    return true;
}();

} // namespace

ColvarsCollectiveVariable::ColvarsCollectiveVariable(const std::string& configString,
                                                     const std::string& colvarName,
                                                     double valueScale)
    : configString_(configString),
      colvarName_(colvarName),
      valueScale_(valueScale),
      pbcMode_(ColvarsForce::PbcMode::Auto)
{
    validate_nonempty(configString_, "configString");
    validate_nonempty(colvarName_, "colvarName");
    setValueScale(valueScale);
    setName("ColvarsCollectiveVariable");
}

const std::string& ColvarsCollectiveVariable::getConfigString() const
{
    return configString_;
}

void ColvarsCollectiveVariable::setConfigString(const std::string& configString)
{
    validate_nonempty(configString, "configString");
    configString_ = configString;
}

const std::string& ColvarsCollectiveVariable::getColvarName() const
{
    return colvarName_;
}

void ColvarsCollectiveVariable::setColvarName(const std::string& colvarName)
{
    validate_nonempty(colvarName, "colvarName");
    colvarName_ = colvarName;
}

double ColvarsCollectiveVariable::getValueScale() const
{
    return valueScale_;
}

void ColvarsCollectiveVariable::setValueScale(double valueScale)
{
    if (!std::isfinite(valueScale)) {
        throw OpenMMException("ColvarsCollectiveVariable: valueScale must be finite");
    }
    valueScale_ = valueScale;
}

bool ColvarsCollectiveVariable::usesPeriodicBoundaryConditions() const
{
    return pbcMode_ == ColvarsForce::PbcMode::Enabled;
}

void ColvarsCollectiveVariable::setUsesPeriodicBoundaryConditions(bool usesPbc)
{
    pbcMode_ = usesPbc ? ColvarsForce::PbcMode::Enabled : ColvarsForce::PbcMode::Disabled;
}

ColvarsForce::PbcMode ColvarsCollectiveVariable::getPbcMode() const
{
    return pbcMode_;
}

void ColvarsCollectiveVariable::setPbcMode(ColvarsForce::PbcMode mode)
{
    switch (mode) {
        case ColvarsForce::PbcMode::Auto:
        case ColvarsForce::PbcMode::Enabled:
        case ColvarsForce::PbcMode::Disabled:
            pbcMode_ = mode;
            return;
    }
    throw OpenMMException("Invalid ColvarsCollectiveVariable PBC mode");
}

ForceImpl* ColvarsCollectiveVariable::createImpl() const
{
    return new ColvarsCollectiveVariableImpl(*this);
}

ColvarsCollectiveVariableImpl::ColvarsCollectiveVariableImpl(
    const ColvarsCollectiveVariable& owner)
    : CustomCPPForceImpl(owner),
      owner_(owner),
      settings_(make_settings(owner))
{
}

ColvarsCollectiveVariableImpl::~ColvarsCollectiveVariableImpl() = default;

const Force& ColvarsCollectiveVariableImpl::getOwner() const
{
    return owner_;
}

void ColvarsCollectiveVariableImpl::initialize(ContextImpl& context)
{
    if (owner_.getForceGroup() == 31) {
        throw OpenMMException(
            "ColvarsCollectiveVariable group 31 is unsupported because this OpenMM "
            "dependency uses a signed force-group shift. Use group 0-30.");
    }
    CustomCPPForceImpl::initialize(context);
    proxy_ = std::make_unique<OpenMMColvarProxy>(
        settings_, context.getSystem(), owner_.getColvarName(), owner_.getValueScale());
}

double ColvarsCollectiveVariableImpl::computeForce(ContextImpl& context,
                                                   const std::vector<Vec3>& positions,
                                                   std::vector<Vec3>& forces)
{
    std::fill(forces.begin(), forces.end(), Vec3());
    const double energy = proxy_->calculate(context, positions);
    proxy_->addForcesToOpenMM(forces);
    return energy;
}

std::map<std::string, double> ColvarsCollectiveVariableImpl::getDefaultParameters()
{
    return {};
}
