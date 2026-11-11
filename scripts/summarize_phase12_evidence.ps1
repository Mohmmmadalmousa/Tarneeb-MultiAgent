# Read saved research only: no Python, checkpoints, games, bootstrap, or training.
# Emits JSON to stdout. Run from the repository root. Never edits source evidence.
$ErrorActionPreference = 'Stop'
$sources = [ordered]@{}
function Read-Evidence([string]$Path) {
    $sources[$Path] = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}
function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
$a1 = Read-Evidence 'results/phase9/a1_report.json'
$a2 = Read-Evidence 'results/phase10/a2_report.json'
$a3 = Read-Evidence 'results/phase11_a3_evaluation_v1/selection.json'
$receipt = Read-Evidence 'results/phase11_freeze_preparation_v1/a3_verification.json'
Require ($a1.seeds.Count -eq 3 -and $a2.seeds.Count -eq 3) 'Expected three training seeds'
$a1Seed = [ordered]@{}
foreach ($anchor in @('random', 'basic', 'strong')) {
    $values = @($a1.per_seed | ForEach-Object { $_.anchors.$anchor.paired_difference.difference_mean })
    $mean = ($values | Measure-Object -Average).Average
    $ss = 0.0
    foreach ($value in $values) { $ss += [Math]::Pow($value - $mean, 2) }
    $sd = [Math]::Sqrt($ss / 2)
    # Two-sided 95% Student t, df=2; same convention as the frozen A2 summary.
    $halfWidth = 4.302652729911275 * $sd / [Math]::Sqrt(3)
    $a1Seed[$anchor] = [ordered]@{n=3; per_seed=$values; mean=$mean; sd=$sd; lower_bound=$mean-$halfWidth; upper_bound=$mean+$halfWidth}
}
$jobs = @()
foreach ($reference in $a3.evidence.PSObject.Properties) {
    $path = 'results/phase11_a3_evaluation_v1/selection/' + $reference.Name + '.json'
    $job = Read-Evidence $path
    $metrics = $job.metrics
    Require ($job.digest -eq $reference.Value -and $job.plan_identity -eq $a3.plan_identity) "Reference mismatch: $path"
    Require ($job.scores.Count -eq $job.job.count -and $job.job.count -eq $metrics.paired_deal_count) "Count mismatch: $path"
    $mean = ($job.scores | Measure-Object -Average).Average
    Require ([Math]::Abs($mean - $metrics.mean_paired_score_difference) -lt 1e-10) "Mean mismatch: $path"
    Require ($metrics.illegal_action_rate -eq 0 -and $metrics.no_trump_action_rate -eq 0) "Safety mismatch: $path"
    Require ($metrics.replay_integrity_rate -eq 1 -and $metrics.replay_verified_count -gt 0) "Replay sample mismatch: $path"
    $jobs += [pscustomobject]@{path=$path; a=$job.job.a; b=$job.job.b; pairs=$metrics.paired_deal_count; metrics=$metrics}
}
$totalPairs = ($jobs | Measure-Object -Property pairs -Sum).Sum
Require ($jobs.Count -eq 99 -and $totalPairs -eq 54000) 'Unexpected A3 coverage'
Require ($sources['results/phase11_a3_evaluation_v1/selection.json'] -eq $receipt.selection_sha256) 'Historical selection receipt mismatch'
Require ($null -eq $a3.selected -and -not $a3.promotion) 'Unexpected selection/promotion'
$pilot = [ordered]@{}
foreach ($opponent in @('parent', 'basic', 'strong', 'champion')) {
    $pilot[$opponent] = Read-Evidence "results/balanced_seating_pilot_v1/evaluation/difference_$opponent.json"
    Require ($pilot[$opponent].pair_count -eq 150) 'Unexpected pilot sample count'
}
[ordered]@{
    scope='Saved evidence only; digests linked, not cryptographically recomputed; historical full provenance audit reused'
    a1_seed_level=$a1Seed
    a2_seed_level=$a2.seed_level
    a3_seed_level=$a3.comparisons
    a3_matchups=$jobs.Count
    a3_duplicate_pairs=$totalPairs
    a3_cross_play=$a3.cross_play
    a3_strong=@($jobs | Where-Object { $_.b -eq 'strong' })
    pilot_balanced_minus_legacy=$pilot
    source_sha256=$sources
} | ConvertTo-Json -Depth 30
