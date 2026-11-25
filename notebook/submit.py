# Hands the agent to the competition. Save & Run All, then submit that version.
from kaggle_evaluation.gga.gga_inference_server import GGAInferenceServer

Path('/kaggle/working/agent.py').write_text('from __main__ import Agent\n')  # the server loads the agent from this file
GGAInferenceServer().run()
