# coding_convention.md

Below are guidelines on how to code.

## Guidelines

- No nested functions.
- All imports should be at the top of the file.
- Follow object oriented design.
- Try to reuse code, don't add duplicate code.
- Maintain separation of concerns.
- Create configs using hydra.
- All outputs of a run should be inside the hydra run directory.
- Don't add fallback methods, raise errors when things fail.
- Don't have large try blocks.
- Use python located in folder ./.venv
- If a task involves running a python code or sh file, please read the logs carefully to see if there are any errors.
- If python code uses hydra, the outputs will be in a hydra output dir. Read the logs from the hydra out dir.

### Project Specific Guideline
- Run the code by using custom/runner/run_proassist_label_generator.sh
- I have hydra configs to change parameters.
- All outputs must be saved inside the hydra out dir.
- Use python from ./.venv

### Important Terminology
- **DST = Dialog State Tracking** (NOT Daylight Saving Time)
- In this project, DST refers to tracking the state of dialogue elements (steps, tasks) in conversational AI, not time zone adjustments