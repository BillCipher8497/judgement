# Tiny exact arithmetic only, no worker process or RNG.
$ErrorActionPreference='Stop'
$costs=@(@(0,4,4,2),@(3,0,2,5),@(0,3,4,1),@(1,4,0,3))
$weights=@(1,2,3,4) # chance probabilities /10; joint action probabilities /4
$rows=[System.Collections.Generic.List[object]]::new()
for($code=0;$code -lt 81;$code++) {
    $digits=@();$rest=$code
    for($k=0;$k -lt 4;$k++) { $digits+=($rest%3);$rest=[int][Math]::Floor($rest/3) }
    $numerator=0
    for($w=0;$w -lt 4;$w++) {
        $one=[int][Math]::Floor($w/2);$two=$w%2
        for($a=0;$a -lt 2;$a++) { for($b=0;$b -lt 2;$b++) {
            $pa=if($a -eq 1){$digits[$one]}else{2-$digits[$one]}
            $pb=if($b -eq 1){$digits[2+$two]}else{2-$digits[2+$two]}
            $numerator+=$weights[$w]*$pa*$pb*$costs[$w][2*$a+$b]
        } }
    }
    $pure=(@($digits|Where-Object {$_ -eq 1}).Count -eq 0)
    $rows.Add(@{code=$code;probability_numerators_over2=$digits;score_numerator_over40=$numerator;pure=$pure})
}
$pureRows=@($rows|Where-Object {$_.pure})
if($rows.Count -ne 81 -or $pureRows.Count -ne 16) { throw 'Enumeration count error' }
$minimum=[int](($pureRows|Measure-Object -Property score_numerator_over40 -Minimum).Minimum)
if($minimum -ne 36 -or @($rows|Where-Object {$_.score_numerator_over40 -lt $minimum}).Count) { throw 'Pure/grid minimum check failed' }
if(($rows|Where-Object {$_.code -eq 40}).score_numerator_over40 -ne 86) { throw 'Uniform hand-calculation mismatch' }
$manual=@(40,64,36,76)
for($u=0;$u -lt 2;$u++) { for($v=0;$v -lt 2;$v++) {
    $subset=@($pureRows|Where-Object {$_.probability_numerators_over2[2] -eq 2*$u -and $_.probability_numerators_over2[3] -eq 2*$v})
    $best=[int](($subset|Measure-Object -Property score_numerator_over40 -Minimum).Minimum)
    if($best -ne $manual[2*$u+$v]) { throw 'Independent four-plan manual check failed' }
} }
$out=Join-Path $PSScriptRoot 'results';New-Item -ItemType Directory -Path $out -Force|Out-Null
$rows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $out 'all_profiles.json')
$result=@{pass=$true;checked_utc=[DateTimeOffset]::UtcNow.ToString('o');worlds=4;pure_plans=16;half_grid_profiles=81;
    pure_minimum='9/10';grid_minimum='9/10';uniform_value='43/20';manual_four_plan_minima=@('1','8/5','9/10','19/10');
    uncertainty='Complete fixed synthetic table/grid, no sampling CI or random seed. Sanity check only; not a Judgement payoff estimate or proof from a sampled grid.'}
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $out 'verification.json')
$result | ConvertTo-Json -Depth 5
