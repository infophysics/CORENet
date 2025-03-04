higgs_mass = 125.09
higgs_mass_sigma = 3.0

dm_relic_density = .11
dm_relic_density_sigma = .03


# Static Definitions
pmssm_parameters = {
    'm1': {
        'bounds': [[-4000.0, -50.0], [50.0, 4000.0]],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'm2': {
        'bounds': [[-4000.0, -100.0], [100.0, 4000.0]],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'm3': {
        'bounds': [400.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mmu': {
        'bounds': [[-4000.0, -100.0], [100.0, 4000.0]],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mA': {
        'bounds': [100.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'At': {
        'bounds': [-4000.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'Ab': {
        'bounds': [-4000.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'Atau': {
        'bounds': [-4000.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mL12': {
        'bounds': [100.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mL3': {
        'bounds': [100.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'me12': {
        'bounds': [100.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'me3': {
        'bounds': [100.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mQ12': {
        'bounds': [400.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mQ3': {
        'bounds': [200.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mu12': {
        'bounds': [400.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mu3': {
        'bounds': [200.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'md12': {
        'bounds': [400.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'md3': {
        'bounds': [200.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'tanb': {
        'bounds': [1.0, 60.0],
        'prior': 'uniform',
        'type': 'continuous'
    }
}

cmssm_parameters = {
    'm1': {
        'bounds': [[0, -50.0], [50.0, 4000.0]],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'm2': {
        'bounds': [[-4000.0, -100.0], [100.0, 4000.0]],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'm3': {
        'bounds': [400.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mmu': {
        'bounds': [[-4000.0, -100.0], [100.0, 4000.0]],
        'prior': 'uniform',
        'type': 'continuous'
    },
    'mA': {
        'bounds': [100.0, 4000.0],
        'prior': 'uniform',
        'type': 'continuous'
    },
}

cmssm_columns = [
    'gut_m0', 'gut_m12', 'gut_A0', 'gut_tanb', 'sign_mu'
]
pmssm_columns = [
    'gut_m1', 'gut_m2', 'gut_m3', 'gut_mmu', 'gut_mA',
    'gut_Ab', 'gut_At', 'gut_Atau', 'gut_mL1', 'gut_mL3',
    'gut_me1', 'gut_mtau1', 'gut_mQ1', 'gut_mQ3', 'gut_mu1',
    'gut_mu3', 'gut_md1', 'gut_md3', 'gut_tanb'
]
weak_soft_columns = [
    'weak_Au', 'weak_Ac', 'weak_At', 'weak_Ad', 'weak_As', 'weak_Ab', 'weak_Ae', 'weak_Amu',
    'weak_Atau', 'weak_tanb', 'weak_gprime', 'weak_g2', 'weak_g3', 'weak_m1', 'weak_m2',
    'weak_m3', 'weak_mA2', 'weak_mmu', 'weak_mH12', 'weak_mH22', 'weak_muR', 'weak_mcR',
    'weak_mtR', 'weak_mdR', 'weak_msR', 'weak_mbR', 'weak_eR', 'weak_mmuR', 'weak_mtauR',
    'weak_mQ1', 'weak_mQ2', 'weak_mQ3', 'weak_mL1', 'weak_mL2', 'weak_mL3', 'weak_higgsvev',
    'weak_Yt', 'weak_Yb', 'weak_Ytau'
]
weak_mass_columns = [
    'weakm_mW', 'weakm_mh', 'weakm_mH', 'weakm_mA', 'weakm_mHpm', 'weakm_m3', 'weakm_mneut1',
    'weakm_mneut2', 'weakm_mneut3', 'weakm_mneut4', 'weakm_mcharg1', 'weakm_mcharg2', 'weakm_mdL',
    'weakm_muL', 'weakm_msL', 'weakm_mcL', 'weakm_mb1', 'weakm_mt1', 'weakm_meL', 'weakm_mesneuL',
    'weakm_mmuL', 'weakm_mmusneuL', 'weakm_mtau1', 'weakm_mtausneuL', 'weakm_mdR', 'weakm_muR',
    'weakm_msR', 'weakm_mcR', 'weakm_mb2', 'weakm_mt2', 'weakm_meR', 'weakm_mmuR', 'weakm_mtau2',
    'weakm_neutmix11', 'weakm_neutmix12', 'weakm_neutmix13', 'weakm_neutmix14'
]
weak_measurements_columns = [
    'omegah2', 'g-2', 'b->sgamma', 'b->sgammaSM', 'B+->taunu', 'Bs->mumu', 'Ds->taunu',
    'Ds->munu', 'deltarho', 'RL23', 'lspmass', 'sigmav', 'cdmpSI', 'cdmpSD', 'cdmnSI', 'cdmnSD'
]
dm_channels_columns = [
    'chan1weight', 'chan1part1', 'chan1part2', 'chan1part3', 'chan1part4',
    'chan2weight', 'chan2part1', 'chan2part2', 'chan2part3', 'chan2part4',
    'chan3weight', 'chan3part1', 'chan3part2', 'chan3part3', 'chan3part4',
    'chan4weight', 'chan4part1', 'chan4part2', 'chan4part3', 'chan4part4',
    'chan5weight', 'chan5part1', 'chan5part2', 'chan5part3', 'chan5part4',
    'chan6weight', 'chan6part1', 'chan6part2', 'chan6part3', 'chan6part4',
    'chan7weight', 'chan7part1', 'chan7part2', 'chan7part3', 'chan7part4',
    'chan8weight', 'chan8part1', 'chan8part2', 'chan8part3', 'chan8part4',
    'chan9weight', 'chan9part1', 'chan9part2', 'chan9part3', 'chan9part4',
    'chan10weight', 'chan10part1', 'chan10part2', 'chan10part3', 'chan10part4'
]
gut_gauge_columns = ['gut_energscale', 'gut_gprime', 'gut_g2', 'gut_g3']

assert (len(cmssm_columns) == 5), "cmssm_columns wrong entries"
assert (len(pmssm_columns) == 19), "pmssm_columns wrong entries"
assert (len(weak_soft_columns) == 39), "weak_soft_columns wrong entries"
assert (len(weak_mass_columns) == 37), "weak_mass_columns wrong entries"
assert (len(weak_measurements_columns) == 16), "weak_measurements_columns wrong entries"
assert (len(dm_channels_columns) == 50), "dm_channels_columns wrong entries"
assert (len(gut_gauge_columns) == 4), "gut_gauge_columns wrong entries"

common_columns = weak_soft_columns + weak_mass_columns + weak_measurements_columns + dm_channels_columns + gut_gauge_columns

# here we have the specification of
# different softsusy output values
# and their corresponding block
softsusy_physical_parameters = {
    # parameter: block
    "Au":   "au",
    "Ac":   "au",
    "At":   "au",
    "Ad":   "ad",
    "As":   "ad",
    "Ab":   "ad",
    "Ae":   "ae",
    "Amu":  "ae",
    "Atau": "ae",
    "tan":  "hmix",
    "g":    "gauge",
    "g'":   "gauge",
    "g3":   "gauge",
    "M_1":  "msoft",
    "M_2":  "msoft",
    "M_3":  "msoft",
    "mA^2": "hmix",
    "mu":   "hmix",
    "mH1^2": "msoft",
    "mH2^2": "msoft",
    "muR":  "msoft",
    "mcR":  "msoft",
    "mtR":  "msoft",
    "mdR":  "msoft",
    "msR":  "msoft",
    "mbR":  "msoft",
    "meR":  "msoft",
    "mmuR": "msoft",
    "mtauR": "msoft",
    "mqL1": "msoft",
    "mqL2": "msoft",
    "mqL3": "msoft",
    "meL":  "msoft",
    "mmuL": "msoft",
    "mtauL": "msoft",
    "higgs": "hmix",
    "Yt":   "yu",
    "Yb":   "yd",
    "Ytau": "ye",
}
# incidentally, all the weak
# parameters are in the MASS block
softsusy_weak_parameters = {
    # parameter: block
    "MW":   "MASS",
    "h0":   "MASS",
    "H0":   "MASS",
    "A0":   "MASS",
    "H+":   "MASS",
    "~g":   "MASS",
    "~neutralino(1)":   "MASS",
    "~neutralino(2)":   "MASS",
    "~neutralino(3)":   "MASS",
    "~neutralino(4)":   "MASS",
    "~chargino(1)":     "MASS",
    "~chargino(2)":     "MASS",
    "~d_L":  "MASS",
    "~u_L":  "MASS",
    "~s_L":  "MASS",
    "~c_L":  "MASS",
    "~b_1":  "MASS",
    "~t_1":  "MASS",
    "~e_L":  "MASS",
    "~nue_L": "MASS",
    "~mu_L": "MASS",
    "~numu_L":   "MASS",
    "~stau_1":   "MASS",
    "~nu_tau_L":  "MASS",
    "~d_R":  "MASS",
    "~u_R":  "MASS",
    "~s_R":  "MASS",
    "~c_R":  "MASS",
    "~b_2":  "MASS",
    "~t_2":  "MASS",
    "~e_R":  "MASS",
    "~mu_R": "MASS",
    "~stau_2":   "MASS",
    "N_{1,1}":   "nmix",
    "N_{1,2}":   "nmix",
    "N_{1,3}":   "nmix",
    "N_{1,4}":   "nmix",
}


def constraints_cmssm(X):
    # check if the relic density is saturated
    relic_density_saturated = (
        (round(X['omegah2'], 2) - dm_relic_density).abs() < dm_relic_density_sigma
    )
    X = X[relic_density_saturated]
    # see if the higgs mass is within experimental bounds
    higgs = (
        (round(X['weakm_mh'], 2) - higgs_mass).abs() < higgs_mass_sigma
    )
    X = X[higgs]
    return X


def constraints_pmssm(X):
    neutralino_lsp = X['lspmass'] == X['weakm_mneut1']
    relic_density_saturated = (X['omegah2'] - dm_relic_density).abs() < dm_relic_density_sigma
    higgs = (X['weakm_mh'] - higgs_mass).abs() < higgs_mass_sigma
    X = X[neutralino_lsp]
    X = X[relic_density_saturated]
    X = X[higgs]
    return X
