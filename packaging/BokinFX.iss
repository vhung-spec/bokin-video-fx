; Inno Setup script for Bokin FX: dist\BokinFX\ (PyInstaller) -> BokinFX-Setup-<version>.exe
; Per-user install, no admin rights: %LOCALAPPDATA%\Programs\Bokin FX, Start menu and desktop shortcuts, an entry
; in Settings > Apps. Projects live in %APPDATA%\Bokin\FX and finished videos in Videos\Bokin FX, so updates and
; uninstalls keep them.
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
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=yes
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

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Code: Integer;
  Installed: String;
begin
  Result := '';
  Installed := ExpandConstant('{app}\{#AppExe}');
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
