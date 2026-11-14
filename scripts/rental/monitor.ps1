param(
    [ValidateSet('train', 'gpu', 'status')]
    [string]$View = 'train',
    [string]$IdentityFile = (Join-Path $env:USERPROFILE '.ssh\id_ed25519')
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$activePath = Join-Path $projectRoot 'logs/training_active.json'
$rental = Get-Content -LiteralPath $activePath -Raw | ConvertFrom-Json
if ($rental.actual_status -eq 'destroyed') {
    throw 'The recorded rental was destroyed. Configure a new active endpoint before monitoring.'
}
if (-not (Test-Path -LiteralPath $identityFile -PathType Leaf)) {
    throw "Existing SSH identity not found: $identityFile. Specify your key with -IdentityFile."
}
$connection = @('-i', $identityFile, '-p', [string]$rental.port,
    '-o', 'IdentitiesOnly=yes', '-o', "UserKnownHostsFile=$($rental.known_hosts)",
    '-o', 'StrictHostKeyChecking=yes')
if ($View -eq 'status') {
    $remoteOutput = if ($rental.remote_output_dir) { [string]$rental.remote_output_dir } else { '/workspace/grpo-mt/logs/production' }
    if ($remoteOutput -notmatch '^/[A-Za-z0-9_./-]+$') { throw 'Unexpected remote output path in rental state.' }
    & ssh @connection $rental.host "if test -f '$remoteOutput/cycle_state.json'; then cat '$remoteOutput/cycle_state.json'; cat '$remoteOutput/latest.json' 2>/dev/null; else cat '$remoteOutput/state.json'; cat '$remoteOutput/sft/result.json' 2>/dev/null || cat '$remoteOutput/sft/progress.json' 2>/dev/null; cat '$remoteOutput/rl/result.json' 2>/dev/null || cat '$remoteOutput/rl/progress.json' 2>/dev/null; fi; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader"
} else {
    $session = if ($View -eq 'gpu') { 'mt-monitor' } elseif ($rental.train_session) { [string]$rental.train_session } else { 'mt-train' }
    if ($session -notmatch '^[A-Za-z0-9_-]+$') { throw 'Unexpected tmux session in rental state.' }
    & ssh @connection -t $rental.host "tmux attach -t $session"
}
exit $LASTEXITCODE
