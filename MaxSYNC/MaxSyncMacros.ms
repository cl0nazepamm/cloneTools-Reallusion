-- MAXsync Macroscripts for 3ds Max
-- Run this once to register. Then find in: Customize > Customize User Interface > Category: CloneTools

-- Resolved at load time so macroscripts can find MaxSyncListener.ms regardless of install location.
global MaxSYNC
global MaxSYNC_MacrosDir
global MaxSYNC_LastLoadError

fn MaxSYNC_isInstallDir dir =
(
	if dir == undefined or dir == "" do return false

	local macrosPath = dir + "MaxSyncMacros.ms"
	local listenerPath = dir + "MaxSyncListener.ms"
	local listenerPyPath = dir + "MaxSyncListenerPy.py"
	(doesFileExist macrosPath) and (doesFileExist listenerPath) and (doesFileExist listenerPyPath)
)

fn MaxSYNC_uniqueAppend arr value =
(
	if value == undefined or value == "" do return arr

	local valueLower = toLower value
	for existing in arr where (toLower existing) == valueLower do return arr
	append arr value
	arr
)

fn MaxSYNC_detectMacrosDir =
(
	local candidateDirs = #()

	if MaxSYNC_MacrosDir != undefined and MaxSYNC_MacrosDir != "" do
		MaxSYNC_uniqueAppend candidateDirs MaxSYNC_MacrosDir

	local sourcePathA = try(getSourceFileName())catch("")
	if sourcePathA != undefined and sourcePathA != "" do
		MaxSYNC_uniqueAppend candidateDirs (getFilenamePath sourcePathA)

	local sourcePathB = try(sourceFileName)catch("")
	if sourcePathB != undefined and sourcePathB != "" do
		MaxSYNC_uniqueAppend candidateDirs (getFilenamePath sourcePathB)

	local currentDir = try(sysInfo.currentdir)catch("")
	if currentDir != undefined and currentDir != "" do
		MaxSYNC_uniqueAppend candidateDirs currentDir

	MaxSYNC_uniqueAppend candidateDirs ((getDir #userScripts) + "\\CloneTools\\")

	for dir in candidateDirs where MaxSYNC_isInstallDir dir do return dir

	undefined
)

fn MaxSYNC_collectMacroPaths =
(
	local paths = #()
	local detectedDir = MaxSYNC_detectMacrosDir()
	if detectedDir != undefined and detectedDir != "" do
		MaxSYNC_uniqueAppend paths (detectedDir + "MaxSyncMacros.ms")

	MaxSYNC_uniqueAppend paths ((getDir #userScripts) + "\\CloneTools\\MaxSyncMacros.ms")
	paths
)

fn MaxSYNC_collectListenerPaths =
(
	local paths = #()
	local detectedDir = MaxSYNC_detectMacrosDir()
	if detectedDir != undefined and detectedDir != "" do
		MaxSYNC_uniqueAppend paths (detectedDir + "MaxSyncListener.ms")

	MaxSYNC_uniqueAppend paths ((getDir #userScripts) + "\\CloneTools\\MaxSyncListener.ms")
	paths
)

fn MaxSYNC_resolveListenerPath =
(
	local paths = MaxSYNC_collectListenerPaths()
	for p in paths where doesFileExist p do return p

	if paths.count > 0 then paths[1] else ""
)

fn MaxSYNC_autoStart =
(
	if ::MaxSYNC == undefined do (
		local scriptPath = MaxSYNC_resolveListenerPath()
		if scriptPath != "" and doesFileExist scriptPath then (
			try (
				MaxSYNC_LastLoadError = ""
				fileIn scriptPath
			) catch (
				MaxSYNC_LastLoadError = "Load failed: " + (getCurrentException() as string) + "\nPath:\n" + scriptPath
				return false
			)

			if ::MaxSYNC == undefined then (
				MaxSYNC_LastLoadError = "MaxSyncListener.ms loaded but global MaxSYNC was not created.\nPath:\n" + scriptPath
				return false
			)
		) else (
			local lookedIn = ""
			for p in (MaxSYNC_collectListenerPaths()) do lookedIn += (p + "\n")
			MaxSYNC_LastLoadError = "MaxSyncListener.ms not found.\n\nLooked in:\n" + lookedIn
			messageBox ("MaxSyncListener.ms not found.\n\nLooked in:\n" + lookedIn) title:"MAXsync Error"
			return false
		)
	)
	::MaxSYNC != undefined
)

macroscript MaxSYNC_Start
	category:"CloneTools"
	toolTip:"MAXsync Start"
	buttonText:"MAXsync Start"
(
	local started = try(MaxSYNC_autoStart())catch(
		local fallbackPath = (getDir #userScripts) + "\\CloneTools\\MaxSyncListener.ms"
		if doesFileExist fallbackPath do (
			try(fileIn fallbackPath)catch(
				MaxSYNC_LastLoadError = "Fallback load failed: " + (getCurrentException() as string) + "\nPath:\n" + fallbackPath
			)
		)
		::MaxSYNC != undefined
	)

	if started then (
		format "[MAXsync] Listener ready.\n"
	) else (
		messageBox "MAXsync could not start listener.\n\nRun MaxSyncMacros.ms once to reinstall scripts and startup loader." title:"MAXsync Error"
		format "[MAXsync] Failed to start listener.\n"
	)
)

macroscript MaxSYNC_Stop
	category:"CloneTools"
	toolTip:"MAXsync Stop"
	buttonText:"MAXsync Stop"
(
	if ::MaxSYNC != undefined then (
		::MaxSYNC.Stop()
	) else (
		format "[MAXsync] Not running.\n"
	)
)

macroscript MaxSYNC_SendProp
	category:"CloneTools"
	toolTip:"Send Prop to iClone"
	buttonText:"Send to iClone"
(
	local started = try(MaxSYNC_autoStart())catch(
		local fallbackPath = (getDir #userScripts) + "\\CloneTools\\MaxSyncListener.ms"
		if doesFileExist fallbackPath do (
			try(fileIn fallbackPath)catch(
				MaxSYNC_LastLoadError = "Fallback load failed: " + (getCurrentException() as string) + "\nPath:\n" + fallbackPath
			)
		)
		::MaxSYNC != undefined
	)

	if started then (
		::MaxSYNC.SendProp()
	) else (
		local errMsg = "MAXsync listener is not available.\n\nExpected file:\n" + ((getDir #userScripts) + "\\CloneTools\\MaxSyncListener.ms")
		if MaxSYNC_LastLoadError != undefined and MaxSYNC_LastLoadError != "" do
			errMsg += ("\n\nDetails:\n" + MaxSYNC_LastLoadError)
		errMsg += "\n\nRun MaxSyncMacros.ms once to reinstall."
		messageBox errMsg title:"MAXsync Error"
	)
)

macroscript MaxSYNC_SendSingleFrame
	category:"CloneTools"
	toolTip:"Send Single Frame to iClone"
	buttonText:"Send Single Frame"
(
	local started = try(MaxSYNC_autoStart())catch(
		local fallbackPath = (getDir #userScripts) + "\\CloneTools\\MaxSyncListener.ms"
		if doesFileExist fallbackPath do (
			try(fileIn fallbackPath)catch(
				MaxSYNC_LastLoadError = "Fallback load failed: " + (getCurrentException() as string) + "\nPath:\n" + fallbackPath
			)
		)
		::MaxSYNC != undefined
	)

	if started then (
		::MaxSYNC.SendSingleFrame()
	) else (
		local errMsg = "MAXsync listener is not available.\n\nExpected file:\n" + ((getDir #userScripts) + "\\CloneTools\\MaxSyncListener.ms")
		if MaxSYNC_LastLoadError != undefined and MaxSYNC_LastLoadError != "" do
			errMsg += ("\n\nDetails:\n" + MaxSYNC_LastLoadError)
		errMsg += "\n\nRun MaxSyncMacros.ms once to reinstall."
		messageBox errMsg title:"MAXsync Error"
	)
)

fn MaxSYNC_copyScriptFile sourcePath targetPath =
(
	if sourcePath == undefined or sourcePath == "" do return false
	if targetPath == undefined or targetPath == "" do return false
	if not doesFileExist sourcePath do return false

	if (toLower sourcePath) == (toLower targetPath) do return true

	if doesFileExist targetPath do deleteFile targetPath
	copyFile sourcePath targetPath
)

fn MaxSYNC_InstallUserScripts =
(
	local sourceDir = MaxSYNC_detectMacrosDir()
	if sourceDir == undefined or sourceDir == "" then (
		format "[MAXsync] Could not detect source folder for installation.\n"
		return false
	)

	local userCloneToolsDir = (getDir #userScripts) + "\\CloneTools\\"
	try(makeDir userCloneToolsDir)catch()

	local sourceMacros = sourceDir + "MaxSyncMacros.ms"
	local sourceListener = sourceDir + "MaxSyncListener.ms"
	local sourceListenerPy = sourceDir + "MaxSyncListenerPy.py"
	local targetMacros = userCloneToolsDir + "MaxSyncMacros.ms"
	local targetListener = userCloneToolsDir + "MaxSyncListener.ms"
	local targetListenerPy = userCloneToolsDir + "MaxSyncListenerPy.py"

	local copiedMacros = MaxSYNC_copyScriptFile sourceMacros targetMacros
	local copiedListener = MaxSYNC_copyScriptFile sourceListener targetListener
	local copiedListenerPy = MaxSYNC_copyScriptFile sourceListenerPy targetListenerPy

	if copiedMacros and copiedListener and copiedListenerPy then (
		MaxSYNC_MacrosDir = userCloneToolsDir
		format "[MAXsync] Installed scripts to: %\n" userCloneToolsDir
		true
	) else (
		format "[MAXsync] Failed to copy scripts to user scripts folder.\n"
		false
	)
)

fn MaxSYNC_escapePathForLiteral path =
(
	substituteString path "\\" "\\\\"
)

fn MaxSYNC_collectStartupDirs =
(
	local dirs = #()

	local userStartupA = try(getDir #userStartupScripts)catch("")
	if userStartupA != undefined and userStartupA != "" do MaxSYNC_uniqueAppend dirs userStartupA

	local userStartupB = (getDir #userScripts) + "\\startup"
	MaxSYNC_uniqueAppend dirs userStartupB

	local startupGlobal = try(getDir #startupScripts)catch("")
	if startupGlobal != undefined and startupGlobal != "" do MaxSYNC_uniqueAppend dirs startupGlobal

	dirs
)

fn MaxSYNC_canWriteDir dirPath =
(
	if dirPath == undefined or dirPath == "" do return false
	try (
		makeDir dirPath
		local probePath = dirPath + "\\_maxsync_write_test.tmp"
		local f = createFile probePath
		close f
		deleteFile probePath
		true
	) catch (
		false
	)
)

fn MaxSYNC_InstallUserMacroMirror =
(
	local sourceDir = MaxSYNC_detectMacrosDir()
	if sourceDir == undefined or sourceDir == "" do return false

	local sourceMacros = sourceDir + "MaxSyncMacros.ms"
	if not doesFileExist sourceMacros do return false

	local userMacrosDir = getDir #userMacros
	if userMacrosDir == undefined or userMacrosDir == "" do return false
	if not MaxSYNC_canWriteDir userMacrosDir do return false

	local mirrorPath = userMacrosDir + "\\MAXsync_CloneTools.mcr"
	if doesFileExist mirrorPath do deleteFile mirrorPath
	local ok = copyFile sourceMacros mirrorPath
	if ok then format "[MAXsync] User macro mirror installed: %\n" mirrorPath
	ok
)

-- One-time installer: drop a startup loader so macros auto-load every 3ds Max boot.
fn MaxSYNC_InstallStartupLoader =
(
	local macroPaths = MaxSYNC_collectMacroPaths()
	if macroPaths.count == 0 do append macroPaths ((getDir #userScripts) + "\\CloneTools\\MaxSyncMacros.ms")

	local macrosListLiteral = ""
	for i = 1 to macroPaths.count do (
		local comma = if i < macroPaths.count then "," else ""
		macrosListLiteral += ("\t\t\"" + (MaxSYNC_escapePathForLiteral macroPaths[i]) + "\"" + comma + "\n")
	)

	local loaderScript =
		"-- Auto-generated by MAXsync\n" +
		"(\n" +
		"\tlocal macroPaths = #(\n" +
		macrosListLiteral +
		"\t)\n" +
		"\tfor p in macroPaths do (\n" +
		"\t\tif doesFileExist p then (\n" +
		"\t\t\tfileIn p quiet:true\n" +
		"\t\t\texit\n" +
		"\t\t)\n" +
		"\t)\n" +
		")\n"

	local installCount = 0
	local startupDirs = MaxSYNC_collectStartupDirs()
	for startupDir in startupDirs do (
		if MaxSYNC_canWriteDir startupDir then (
			local loaderPath = startupDir + "\\MAXsync_LoadMacros.ms"
			try (
				local f = createFile loaderPath
				format "%" loaderScript to:f
				close f
				installCount += 1
				format "[MAXsync] Startup loader installed: %\n" loaderPath
			) catch ()
		)
	)

	if installCount > 0 then (
		true
	) else (
		format "[MAXsync] Failed to install startup loader in writable startup folders.\n"
		false
	)
)

-- Capture our own directory so macroscripts can locate MaxSyncListener.ms at runtime.
(
	local selfPath = getSourceFileName()
	if selfPath != undefined and selfPath != "" do
		MaxSYNC_MacrosDir = getFilenamePath selfPath
)

MaxSYNC_InstallUserScripts()
MaxSYNC_InstallUserMacroMirror()
MaxSYNC_InstallStartupLoader()
