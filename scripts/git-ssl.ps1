# Run git with SSL verification disabled (session only — does not change git config).
# Usage: .\scripts\git-ssl.ps1 push origin main
$env:GIT_SSL_NO_VERIFY = 'true'
& git @args
exit $LASTEXITCODE
