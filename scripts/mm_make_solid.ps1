param([string]$Action='probe')
$ErrorActionPreference='Stop'
$trialRoot='D:\WorkSpace\Dental\multiview_teeth_reconstruction\runs\dmm_make_solid_00OMSZGW_20261017_01'
$apiRoot=Join-Path $trialRoot 'mm-api\distrib\csharp_win\Release'
$env:PATH=$apiRoot+';'+$env:PATH
$assemblyPath=Join-Path $apiRoot 'mmapi_csharp.dll'
$null=[System.Reflection.Assembly]::LoadFrom($assemblyPath)
$source=Get-Content -Raw (Join-Path $apiRoot 'RemoteControl.cs')
$source=$source.Replace('class RemoteControl','public class RemoteControl').Replace('m_pReceiveSock = new UdpClient(  );','m_pReceiveSock = new UdpClient(  ); m_pReceiveSock.Client.ReceiveTimeout = 60000;')
Add-Type -TypeDefinition $source -ReferencedAssemblies @($assemblyPath,'System.Core')
$remote=New-Object mmapi_util.RemoteControl
$null=$remote.Initialize()
try {
    if ($Action -eq 'solid') {
        $outPath='D:\WorkSpace\Dental\multiview_teeth_reconstruction\runs\mm_solid17.stl'
        if(Test-Path -LiteralPath $outPath){throw 'Output exists; refusing overwrite'}
        $ids=New-Object vectori
        $ids.Add([int](Get-Content (Join-Path $trialRoot 'owned_object.txt')))
        $cmd=New-Object StoredCommands
        $null=$cmd.AppendSceneCommand_SelectObjects($ids)
        $null=$remote.ExecuteCommands($cmd)
        $cmd=New-Object StoredCommands
        $null=$cmd.AppendBeginToolCommand('makeSolid')
        $null=$remote.ExecuteCommands($cmd)
        Write-Output 'Make Solid started'
        $cmd=New-Object StoredCommands
        $null=$cmd.AppendToolParameterCommand('solidType',[int]2)
        $null=$cmd.AppendToolParameterCommand('solidResolution',[int]512)
        $null=$cmd.AppendToolParameterCommand('meshResolution',[int]512)
        $null=$cmd.AppendToolParameterCommand('offsetDistance',[single]0)
        $null=$cmd.AppendToolParameterCommand('minThickness',[single]0)
        $null=$cmd.AppendToolParameterCommand('closeHoles',[bool]$true)
        $null=$cmd.AppendToolParameterCommand('autoRepair',[bool]$true)
        $null=$remote.ExecuteCommands($cmd)
        $cmd=New-Object StoredCommands
        $null=$cmd.AppendToolUtilityCommand('update')
        $null=$remote.ExecuteCommands($cmd)
        Write-Output 'Make Solid updated'
        foreach($name in @('solidType','solidResolution','meshResolution','solidCellSize','meshCellSize','offsetDistanceWorld','minThicknessWorld','closeHoles','autoRepair')) {
            $cmd=New-Object StoredCommands
            $key=$cmd.AppendGetToolParameterCommand($name)
            $null=$remote.ExecuteCommands($cmd)
            $r=New-Object any_result
            $ok=$cmd.GetToolParameterCommandResult($key,$r)
            Write-Output ($name+': found='+$ok+' type='+$r.type+' f='+$r.f+' i='+$r.i+' b='+$r.b)
        }
        $cmd=New-Object StoredCommands
        $null=$cmd.AppendCompleteToolCommand('accept')
        $null=$remote.ExecuteCommands($cmd)
        $cmd=New-Object StoredCommands
        $key=$cmd.AppendSceneCommand_ListSelectedObjects()
        $null=$remote.ExecuteCommands($cmd)
        $sel=New-Object vectori
        $ok=$cmd.GetSceneCommandResult_ListObjects($key,$sel)
        Write-Output ('Accepted selected='+$sel.Count)
        if(-not $ok -or $sel.Count -ne 1){throw 'Ambiguous result selection; no export'}
        $cmd=New-Object StoredCommands
        $null=$cmd.AppendSceneCommand_ExportMeshFile_CurrentSelection($outPath)
        $null=$remote.ExecuteCommands($cmd)
        Write-Output ('Exported '+$outPath)
    }
    if ($Action -eq 'import') {
        $cmd=New-Object StoredCommands
        $key=$cmd.AppendSceneCommand_AppendMeshFile('D:\WorkSpace\Dental\multiview_teeth_reconstruction\runs\dmm_proxy_plane_00OMSZGW_20261016_01\proxy_ONLY_inside_outside.obj')
        $null=$remote.ExecuteCommands($cmd)
        $ids=New-Object vectori
        $ok=$cmd.GetSceneCommandResult_AppendMeshFile($key,$ids)
        Write-Output ('Import result='+$ok+' count='+$ids.Count)
        foreach($id in $ids){Write-Output ('ID='+$id)}
        if(-not $ok -or $ids.Count -ne 1){throw 'Import did not return exactly one owned object; stop'}
        [System.IO.File]::WriteAllText((Join-Path $trialRoot 'owned_object.txt'),[string]$ids[0])
    }
    if ($Action -eq 'probe') {
        $cmd=New-Object StoredCommands
        $key=$cmd.AppendSceneCommand_ListObjects()
        $null=$remote.ExecuteCommands($cmd)
        $ids=New-Object vectori
        $ok=$cmd.GetSceneCommandResult_ListObjects($key,$ids)
        Write-Output ('API connected='+$ok+' objectCount='+$ids.Count)
        [StoredCommands].GetMethods() | Where-Object { $_.Name -match 'Append.*(Import|Export|SetToolParameter|SelectObjects)|GetToolParameter|GetSceneCommandResult_Append' } | ForEach-Object { $_.ToString() }
    }
} catch { $detail=$_.Exception; while ($null -ne $detail) { Write-Output $detail.ToString(); $detail=$detail.InnerException }; throw } finally { $remote.Shutdown() }
