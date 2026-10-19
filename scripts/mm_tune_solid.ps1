param([string]$Tag='s1024_m512',[int]$SolidResolution=1024,[int]$MeshResolution=512,[int]$SolidType=2,[bool]$AutoRepair=$true,[string]$RunName='dmm_ms_tune_00OMSZGW_20261017_01',[string]$InputSource='runs/dmm_proxy_plane_00OMSZGW_20261016_01/proxy_ONLY_inside_outside.obj')
$ErrorActionPreference='Stop'
$workspace='D:\WorkSpace\Dental\multiview_teeth_reconstruction'
$trialRoot=Join-Path $workspace ('runs\'+$RunName)
$dest=Join-Path $trialRoot $Tag
if(Test-Path -LiteralPath $dest){throw 'Trial already exists; refuse overwrite'}
$null=New-Item -ItemType Directory -Path $dest
$outPath=Join-Path $workspace ('runs\mt_'+$Tag+'.stl')
if(Test-Path -LiteralPath $outPath){throw 'Export exists; refuse overwrite'}
$apiRoot=Join-Path $workspace 'runs\dmm_make_solid_00OMSZGW_20261017_01\mm-api\distrib\csharp_win\Release'
$env:PATH=$apiRoot+';'+$env:PATH
$assemblyPath=Join-Path $apiRoot 'mmapi_csharp.dll'
$null=[System.Reflection.Assembly]::LoadFrom($assemblyPath)
$source=Get-Content -Raw (Join-Path $apiRoot 'RemoteControl.cs')
$source=$source.Replace('class RemoteControl','public class RemoteControl').Replace('m_pReceiveSock = new UdpClient(  );','m_pReceiveSock = new UdpClient(  ); m_pReceiveSock.Client.ReceiveTimeout = 600000;')
Add-Type -TypeDefinition $source -ReferencedAssemblies @($assemblyPath,'System.Core')
$remote=New-Object mmapi_util.RemoteControl
$null=$remote.Initialize()
$record=[ordered]@{tag=$Tag;solidResolution=$SolidResolution;meshResolution=$MeshResolution;solidType=$SolidType;offsetDistance=0;minThickness=0;closeHoles=$true;autoRepair=$AutoRepair;export=$outPath;source=$InputSource;started=(Get-Date -Format o)}
try {
    $cmd=New-Object StoredCommands
    $key=$cmd.AppendSceneCommand_AppendMeshFile((Join-Path $workspace $record.source))
    $null=$remote.ExecuteCommands($cmd)
    $ids=New-Object vectori
    $ok=$cmd.GetSceneCommandResult_AppendMeshFile($key,$ids)
    if(-not $ok -or $ids.Count -ne 1){throw 'Import failed or ambiguous'}
    $record.inputObjectID=$ids[0]
    $cmd=New-Object StoredCommands
    $null=$cmd.AppendSceneCommand_SelectObjects($ids)
    $null=$remote.ExecuteCommands($cmd)
    $cmd=New-Object StoredCommands
    $null=$cmd.AppendBeginToolCommand('makeSolid')
    $null=$remote.ExecuteCommands($cmd)
    Write-Output ($Tag+' started '+(Get-Date -Format o))
    $cmd=New-Object StoredCommands
    $null=$cmd.AppendToolParameterCommand('solidType',[int]$SolidType)
    $null=$cmd.AppendToolParameterCommand('solidResolution',[int]$SolidResolution)
    $null=$cmd.AppendToolParameterCommand('meshResolution',[int]$MeshResolution)
    $null=$cmd.AppendToolParameterCommand('offsetDistance',[single]0)
    $null=$cmd.AppendToolParameterCommand('minThickness',[single]0)
    $null=$cmd.AppendToolParameterCommand('closeHoles',[bool]$true)
    $null=$cmd.AppendToolParameterCommand('autoRepair',[bool]$AutoRepair)
    $null=$remote.ExecuteCommands($cmd)
    $cmd=New-Object StoredCommands
    $null=$cmd.AppendToolUtilityCommand('update')
    $null=$remote.ExecuteCommands($cmd)
    $actual=[ordered]@{}
    foreach($name in @('solidType','solidResolution','meshResolution','solidCellSize','meshCellSize','offsetDistanceWorld','minThicknessWorld','closeHoles','autoRepair')) {
        $cmd=New-Object StoredCommands
        $key=$cmd.AppendGetToolParameterCommand($name)
        $null=$remote.ExecuteCommands($cmd)
        $r=New-Object any_result
        if(-not $cmd.GetToolParameterCommandResult($key,$r)){throw ('Missing parameter '+$name)}
        $value=switch($r.type){0 {$r.f};1 {$r.i};2 {$r.b}}
        $actual[$name]=$value
        Write-Output ($name+'='+$value)
    }
    $record.actual=$actual
    $cmd=New-Object StoredCommands
    $null=$cmd.AppendCompleteToolCommand('accept')
    $null=$remote.ExecuteCommands($cmd)
    $cmd=New-Object StoredCommands
    $key=$cmd.AppendSceneCommand_ListSelectedObjects()
    $null=$remote.ExecuteCommands($cmd)
    $sel=New-Object vectori
    if(-not $cmd.GetSceneCommandResult_ListObjects($key,$sel) -or $sel.Count -ne 1){throw 'Ambiguous output selection'}
    $record.outputObjectID=$sel[0]
    $cmd=New-Object StoredCommands
    $null=$cmd.AppendSceneCommand_ExportMeshFile_CurrentSelection($outPath)
    $null=$remote.ExecuteCommands($cmd)
    if(-not (Test-Path -LiteralPath $outPath)){throw 'Export missing'}
    $record.status='EXPORTED';$record.finished=Get-Date -Format o
    Write-Output ('Exported '+$outPath)
}catch{$record.status='FAILED';$record.error=$_.ToString();throw}
finally{$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $dest 'parameters.json') -Encoding UTF8;$remote.Shutdown()}
