; Inno Setup script for Bokin FX: dist\BokinFX\ (PyInstaller) -> BokinFX-Setup-<version>.exe
; Per-user install, no admin rights, in a folder the user chooses (the owner asked 2026-11-03: "khi cài đặt app có
; thể chọn được nơi chứa app, không mặc định ở ổ C"): the first time Bokin FX on the first fixed drive other than
; Windows' own that the user can write to (D:\Bokin FX), else %LOCALAPPDATA%\Programs\Bokin FX; then where it was
; (updates from the app install there, silently); run by hand, setup offers that folder and another one may be
; chosen: the app moves there (the old copy's files are removed). Start menu and desktop shortcuts, an entry in
; Settings > Apps. Projects live in %APPDATA%\Bokin\FX and finished videos in Videos\Bokin FX, so updates, moves
; and uninstalls keep them.
;   iscc /DAppVersion=1.0.1 packaging\BokinFX.iss
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef DistDir
  #define DistDir "..\dist"
#endif
#define AppName "Bokin FX"
#define AppExe "BokinFX.exe"

[Setup]
; the AppId never changes: a new version installs over the old one
AppId={{10897464-A25C-47D4-89B9-E0110ED62F27}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Bokin
DefaultDirName={code:DefaultDir}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=no
UsePreviousAppDir=yes
PrivilegesRequired=lowest
OutputDir={#DistDir}
OutputBaseFilename=BokinFX-Setup-{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; drop the previous version's libraries before copying the new ones
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#DistDir}\BokinFX\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
; an update started from the app ("Cập nhật ngay" passes /RELAUNCH=1) starts it again; its window reconnects
Filename: "{app}\{#AppExe}"; Parameters: "--updated"; Flags: nowait; Check: Relaunch

[Code]
{ Before files are replaced, the installed app closes a running copy of itself (BokinFX.exe --close). It answers
  3 while that copy is making a video: setup stops then and says why. A copy that does not answer is ended. }
function Relaunch(): Boolean;
begin
  Result := ExpandConstant('{param:relaunch|0}') = '1';
end;

function GetDriveType(lpRootPathName: String): Cardinal;
  external 'GetDriveTypeW@kernel32.dll stdcall';

{ The folder offered the first time: Bokin FX on the first fixed drive other than Windows' own (D:, E: ...) that
  this user can write to; else the user's own programs folder (on Windows' drive, as before). }
function DefaultDir(Param: String): String;
var
  I: Integer;
  Letters, Letter, Root, Probe, Win: String;
begin
  Result := ExpandConstant('{localappdata}\Programs\{#AppName}');
  Win := Uppercase(Copy(ExpandConstant('{win}'), 1, 1));
  Letters := 'DEFGHIJKLMNOPQRSTUVWXYZ';
  for I := 1 to Length(Letters) do
  begin
    Letter := Copy(Letters, I, 1);
    Root := Letter + ':\';
    if (Letter <> Win) and (GetDriveType(Root) = 3) then  { DRIVE_FIXED: not a CD, not a USB stick }
    begin
      Probe := Root + 'bokin-fx-setup-probe';
      if CreateDir(Probe) then
      begin
        RemoveDir(Probe);
        Result := Root + '{#AppName}';
        exit;
      end;
    end;
  end;
end;

{ The folder chosen must be one this user can write to (no admin rights: not Program Files). }
function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir, Probe: String;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    Dir := WizardDirValue;
    Probe := AddBackslash(Dir) + 'bokin-fx-setup-probe.tmp';
    if ForceDirectories(Dir) and SaveStringToFile(Probe, '', False) then
      DeleteFile(Probe)
    else
    begin
      MsgBox('Bokin FX cannot be installed in this folder: you cannot save files there without administrator ' +
        'rights. Choose another folder, for example D:\{#AppName}.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

{ Where the app was installed before, if it was somewhere else than the folder chosen now (it is moving). }
function OldDir(): String;
var
  Old: String;
begin
  Result := '';
  Old := RemoveBackslashUnlessRoot(WizardForm.PrevAppDir);
  if (Old <> '') and (CompareText(Old, RemoveBackslashUnlessRoot(ExpandConstant('{app}'))) <> 0) and
     FileExists(AddBackslash(Old) + '{#AppExe}') then
    Result := Old;
end;

{ Moved: the old copy's own files go (the app, its libraries, its uninstaller; Windows' list of apps now names the
  new folder), and its folder if nothing else is left in it. Projects and videos are elsewhere: they stay. }
procedure CurStepChanged(CurStep: TSetupStep);
var
  Old: String;
begin
  if CurStep <> ssPostInstall then
    exit;
  Old := OldDir();
  if Old = '' then
    exit;
  DeleteFile(AddBackslash(Old) + '{#AppExe}');
  DelTree(AddBackslash(Old) + '_internal', True, True, True);
  DelTree(AddBackslash(Old) + 'unins*.*', False, True, False);
  RemoveDir(Old);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Code: Integer;
  Installed: String;
begin
  Result := '';
  Installed := ExpandConstant('{app}\{#AppExe}');
  if (not FileExists(Installed)) and (OldDir() <> '') then
    Installed := AddBackslash(OldDir()) + '{#AppExe}';  { (moving: the copy running is the old one) }
  if FileExists(Installed) then
  begin
    if Exec(Installed, '--close', '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 3) then
    begin
      Result := 'Bokin FX is still making a video. Wait until it has finished, then run this setup again.';
      exit;
    end;
  end;
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /IM {#AppExe}', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;
