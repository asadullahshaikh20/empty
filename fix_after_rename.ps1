# Run once after renaming/moving the project folder (from the new folder):
#   powershell -ExecutionPolicy Bypass -File .\fix_after_rename.ps1
# The virtual environment stores absolute paths; this re-points it to the new location.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --no-deps -e .\ml                                         # editable ML package
& $py -m pip install --no-deps --force-reinstall uvicorn alembic pytest pip   # re-create .exe launchers
& $py -c "import iqa_ml; print('iqa_ml OK:', iqa_ml.__file__)"
Write-Host "Done. Start the API with: cd backend; ..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000"
