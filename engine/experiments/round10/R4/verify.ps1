$ErrorActionPreference='Stop'
$taskDir=$PSScriptRoot
$engineDir=Split-Path (Split-Path (Split-Path $taskDir))
$protocolPath=Join-Path $taskDir 'PROTOCOL.md'
$protocol=Get-Content -LiteralPath $protocolPath -Raw
$schedule=@()
for($r=0;$r -lt 25;$r++) {
    if($r%5 -eq 4) { $schedule+=@{round_index=$r;n=([Math]::Abs($r-12)+1);dealer=($r%4)} }
}
if(($schedule.n -join ',') -ne '9,4,3,8,13') { throw 'Schedule arithmetic error' }
$rules=Get-Content -LiteralPath (Join-Path $engineDir 'judgement/rules.py') -Raw
foreach($s in @('tuple(range(M, 0, -1)) + tuple(range(2, M + 1))','(0, 1, 2, 3, NT)','self.trump_cycle[r % len(self.trump_cycle)]')) {
    if(-not $rules.Contains($s)) { throw "Rules source assumption mismatch: $s" }
}
foreach($s in @('protocol only','No fitted model','1800M','1810M','1820M','1830M','1840M','full paired game','unknown holdings','candidate-specific','No draw','before screen games','no retuning')) {
    if(-not $protocol.Contains($s)) { throw "Missing protocol constraint: $s" }
}
$out=Join-Path $taskDir 'results';New-Item -ItemType Directory -Path $out -Force|Out-Null
$result=@{pass=$true;checked_utc=[DateTimeOffset]::UtcNow.ToString('o');default_NT_schedule=$schedule;
    protocol_sha256=(Get-FileHash -LiteralPath $protocolPath).Hash;rules_sha256=(Get-FileHash -LiteralPath (Join-Path $engineDir 'judgement/rules.py')).Hash;
    no_simulation_or_seed_draw=$true;limits='Protocol/source/schedule check only. Feature implementation, seed collision check, fitting and performance remain unexecuted.'}
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $out 'verification.json')
$result | ConvertTo-Json -Depth 6
