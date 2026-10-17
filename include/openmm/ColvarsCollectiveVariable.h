#ifndef OPENMM_COLVARS_COLLECTIVE_VARIABLE_H
#define OPENMM_COLVARS_COLLECTIVE_VARIABLE_H

#include "openmm/ColvarsForce.h"
#include "openmm/Force.h"

#include <string>

namespace OpenMM {

/// OpenMM Force whose energy is the value of one Colvars collective variable.
///
/// The energy is the value of the colvar named `colvarName` in the Colvars
/// configuration, multiplied by `valueScale`, and the forces are minus its
/// gradient with respect to the particle positions, multiplied by the same
/// factor. Colvars reports values in its own units (Angstrom for lengths,
/// degrees for angles), so use a `valueScale` of 0.1 for a length in nm or of
/// pi/180 for an angle in radians. The value of a periodic colvar is reported
/// wrapped into its period, so this force is meant to be used as a collective
/// variable of another force, such as a CustomCVForce, which handles the
/// periodicity.
///
/// The configuration may define other colvars, for example to combine them in a
/// customFunction, but no biases, and the reported colvar must be a scalar without
/// an extended Lagrangian. No Colvars output files are written.
///
/// Unlike ColvarsForce, this force can be serialized, so it can be copied and used
/// inside a CustomCVForce.
class ColvarsCollectiveVariable : public Force {
public:
    ColvarsCollectiveVariable(const std::string& configString,
                              const std::string& colvarName,
                              double valueScale = 1.0);

    const std::string& getConfigString() const;
    void setConfigString(const std::string& configString);

    const std::string& getColvarName() const;
    void setColvarName(const std::string& colvarName);

    double getValueScale() const;
    void setValueScale(double valueScale);

    bool usesPeriodicBoundaryConditions() const override;
    void setUsesPeriodicBoundaryConditions(bool usesPbc);

    ColvarsForce::PbcMode getPbcMode() const;
    void setPbcMode(ColvarsForce::PbcMode mode);

protected:
    ForceImpl* createImpl() const override;

private:
    std::string configString_;
    std::string colvarName_;
    double valueScale_;
    ColvarsForce::PbcMode pbcMode_;
};

} // namespace OpenMM

#endif
