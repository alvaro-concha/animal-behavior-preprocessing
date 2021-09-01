import os
from pathlib import Path

repo_path = Path(__file__).parent.parent.absolute().as_posix().replace(" ", r"\ ")
cmd = f"pipreqs {repo_path}"
os.system(cmd)
