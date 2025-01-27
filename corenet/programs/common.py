"""
Standard model inputs for SoftSUSY.
1) alpha^(-1) : inverse electromagnetic coupling
2) G_F        : Fermi constant
3) alpha_s    : strong coupling at the Z pole
4) m_Z        : pole mass
5) m_b        : b quark running mass
6) m_t        : pole mass
7) m_tau      : pole mass
"""
sm_inputs = [
    ["BLOCK SMINPUTS"],
    [" 1 1.27934e+02"],
    [" 2 1.16637e-05"],
    [" 3 1.17200e-01"],
    [" 4 9.11876e+01"],
    [" 5 4.25000e+00"],
    [" 6 1.74200e+02"],
    [" 7 1.77700e+00"],
]


def sugra_input(
    m_scalar: float = 125.0,
    m_gaugino: float = 500.0,
    trilinear: float = 0.0,
    higgs_tanbeta: float = 10.0,
    sign_mu: float = 1.0,
):
    """
    This function generates the input for SoftSUSY.
    The default inputs to this model are from the
    CMSSM10.1.1.
    """
    sugra = [["BLOCK MODSEL"]]
    sugra.append([" 1 1"])          # mSUGRA model
    sugra.append([" 11 32"])        # number of log-spaced grid points
    sugra.append([" 12 10.000e+17"])   # largest Q scale
    sugra += sm_inputs
    sugra.append(["BLOCK MINPAR"])
    sugra.append([f" 3 {higgs_tanbeta:.6e}"])
    sugra.append([f" 4 {int(sign_mu)}"])
    sugra.append([f" 5 {trilinear:.6e}"])
    sugra.append(["BLOCK EXTPAR"])
    # Gaugino masses
    sugra.append([f" 1 {m_gaugino:.6e}"])   # Bino mass
    sugra.append([f" 2 {m_gaugino:.6e}"])   # Wino mass
    sugra.append([f" 3 {m_gaugino:.6e}"])   # gluino mass
    # trilinear couplings
    sugra.append([f" 11 {trilinear:.6e}"])  # Top trilinear coupling
    sugra.append([f" 12 {trilinear:.6e}"])  # Bottom trilinear coupling
    sugra.append([f" 13 {trilinear:.6e}"])  # tau trilinear coupling
    # Higgs parameters
    sugra.append([f" 21 {pow(m_scalar, 2):.6e}"])  # down type higgs mass^2
    sugra.append([f" 22 {pow(m_scalar, 2):.6e}"])  # up type higgs mass^2
    # sfermion masses
    sugra.append([f" 31 {m_scalar:.6e}"])   # left 1st-gen scalar lepton
    sugra.append([f" 32 {m_scalar:.6e}"])   # left 2nd-gen scalar lepton
    sugra.append([f" 33 {m_scalar:.6e}"])   # left 3rd-gen scalar lepton
    sugra.append([f" 34 {m_scalar:.6e}"])   # right scalar electron mass
    sugra.append([f" 35 {m_scalar:.6e}"])   # right scalar muon mass
    sugra.append([f" 36 {m_scalar:.6e}"])   # right scalar tau mass
    sugra.append([f" 41 {m_scalar:.6e}"])   # left 1st-gen scalar quark
    sugra.append([f" 42 {m_scalar:.6e}"])   # left 2nd-gen scalar quark
    sugra.append([f" 43 {m_scalar:.6e}"])   # left 3rd-gen scalar quark
    sugra.append([f" 44 {m_scalar:.6e}"])   # right scalar up
    sugra.append([f" 45 {m_scalar:.6e}"])   # right scalar charm
    sugra.append([f" 46 {m_scalar:.6e}"])   # right scalar top
    sugra.append([f" 47 {m_scalar:.6e}"])   # right scalar down
    sugra.append([f" 48 {m_scalar:.6e}"])   # right scalar strange
    sugra.append([f" 49 {m_scalar:.6e}"])   # right scalar bottom

    return sugra


def universal_input(
    m_bino: float,
    m_wino: float,
    m_gluino: float,
    trilinear_top: float,
    trilinear_bottom: float,
    trilinear_tau: float,
    higgs_mu: float,
    higgs_pseudo: float,
    m_left_electron: float,
    m_left_tau: float,
    m_right_electron: float,
    m_right_tau: float,
    m_scalar_quark1: float,
    m_scalar_quark3: float,
    m_scalar_up: float,
    m_scalar_top: float,
    m_scalar_down: float,
    m_scalar_bottom: float,
    higgs_tanbeta: float,
):
    """
    This function generates the input for SoftSUSY for
    more general theories (including PMSSM).
    """
    universal = [["BLOCK MODSEL"]]
    universal.append([" 1 0"])          # Universal model
    universal.append([" 11 32"])        # number of log-spaced grid points
    universal.append([" 12 1.000e17"])    # largest Q scale
    universal += sm_inputs
    universal.append(["BLOCK MINPAR"])
    universal.append([f" 3 {higgs_tanbeta:.6e}"])
    universal.append(["BLOCK EXTPAR"])
    # input scale
    universal.append([" 0 -1"])    # a priori unknown input scale
    # Gaugino masses
    universal.append([f" 1 {m_bino:.6e}"])      # Bino mass
    universal.append([f" 2 {m_wino:.6e}"])      # Wino mass
    universal.append([f" 3 {m_gluino:.6e}"])    # Gluino mass
    # Trilinear couplings
    universal.append([f" 11 {trilinear_top:.6e}"])      # Top trilinear coupling
    universal.append([f" 12 {trilinear_bottom:.6e}"])   # Bottom trilinear coupling
    universal.append([f" 13 {trilinear_tau:.6e}"])      # Tau trilinear coupling
    # Higgs parameters
    universal.append([f" 23 {higgs_mu:.6e}"])       # mu parameter
    universal.append([f" 26 {higgs_pseudo:.6e}"])   # psuedoscalar higgs pole mass
    # sfermion masses
    universal.append([f" 31 {m_left_electron:.6e}"])    # left 1st-gen scalar lepton
    universal.append([f" 32 {m_left_electron:.6e}"])    # left 2nd-gen scalar lepton
    universal.append([f" 33 {m_left_tau:.6e}"])         # left 3rd-gen scalar lepton
    universal.append([f" 34 {m_right_electron:.6e}"])   # right scalar electron mass
    universal.append([f" 35 {m_right_electron:.6e}"])   # right scalar muon mass
    universal.append([f" 36 {m_right_tau:.6e}"])        # right scalar tau mass
    universal.append([f" 41 {m_scalar_quark1:.6e}"])    # left 1st-gen scalar quark
    universal.append([f" 42 {m_scalar_quark1:.6e}"])    # left 2nd-gen scalar quark
    universal.append([f" 43 {m_scalar_quark3:.6e}"])    # left 3rd-gen scalar quark
    universal.append([f" 44 {m_scalar_up:.6e}"])        # right scalar up
    universal.append([f" 45 {m_scalar_up:.6e}"])        # right scalar charm
    universal.append([f" 46 {m_scalar_top:.6e}"])       # right scalar top
    universal.append([f" 47 {m_scalar_down:.6e}"])      # right scalar down
    universal.append([f" 48 {m_scalar_down:.6e}"])      # right scalar strange
    universal.append([f" 49 {m_scalar_bottom:.6e}"])    # right scalar bottom

    return universal
