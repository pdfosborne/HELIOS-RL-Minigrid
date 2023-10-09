from datetime import datetime
import pandas as pd
# ====== HELIOS IMPORTS =========================================
# ------ Train/Test Function Imports ----------------------------
from helios_rl import STANDARD_RL
#from helios_rl import HELIOS_SEARCH
#from helios_rl import HELIOS_OPTIMIZE
# ------ Config Import ------------------------------------------
# Meta parameters
from helios_rl.config import TestingSetupConfig
# Local parameters
from helios_rl.config_local import ConfigSetup
# ====== LOCAL IMPORTS ==========================================
# ------ Local Environment --------------------------------------
from environment.env import Environment
# ------ Visual Analysis -----------------------------------------------


def main():
    # ------ Load Configs -----------------------------------------
    # Meta parameters
    ExperimentConfig = TestingSetupConfig("./config.json").state_configs
    # Local Parameters
    ProblemConfig = ConfigSetup("./config_local.json").state_configs

    # Specify save dir
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    save_dir = './output/'+str('test')+'_'+time 

    # --------------------------------------------------------------------
    # Flat Baselines
    flat = STANDARD_RL(Config=ExperimentConfig, LocalConfig=ProblemConfig, 
                Environment=Environment,
                save_dir=save_dir, show_figures = 'No', window_size=0.1)
    flat.train()  
    flat.test()
    

if __name__=='__main__':
    main()