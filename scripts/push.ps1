# Push main to GitHub using GIT_SSL_NO_VERIFY only.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:GIT_SSL_NO_VERIFY = 'true'
git push origin main
exit $LASTEXITCODE
