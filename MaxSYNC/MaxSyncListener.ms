-- MAXsync Listener for 3ds Max (v6 - Python socket backend)
-- Keeps the same MaxSYNC API surface while delegating socket I/O to MaxSyncListenerPy.py.
-- MAXsync.Start() / MAXsync.Stop() - listener for iClone -> Max
-- MAXsync.SendProp() / MAXsync.SendSingleFrame() - send selected objects to iClone

(
	global MaxSYNC
	global MaxSYNC_MacrosDir

	struct MaxSYNCStruct (
		port = 29999,
		iClonePort = 29998,
		host = "127.0.0.1",
		protocolVersion = "MSYNC1",
		isRunning = false,
		tempFile = (getDir #temp + "\\MaxSync_ToiClone.fbx"),
		pyScriptPath = "",

		fn log msg = (
			format "[MAXsync] %\n" msg
		),

		fn escPy s = (
			local t = s as string
			t = substituteString t "\\" "\\\\"
			t = substituteString t "'" "\\'"
			t
		),

		fn resolvePyPath = (
			if pyScriptPath != "" and doesFileExist pyScriptPath do return pyScriptPath

			local candidates = #()
			local sourcePath = try(getSourceFileName())catch("")
			if sourcePath != undefined and sourcePath != "" do append candidates ((getFilenamePath sourcePath) + "MaxSyncListenerPy.py")
			if MaxSYNC_MacrosDir != undefined and MaxSYNC_MacrosDir != "" do append candidates (MaxSYNC_MacrosDir + "MaxSyncListenerPy.py")
			append candidates ((getDir #userScripts) + "\\CloneTools\\MaxSyncListenerPy.py")

			for p in candidates where doesFileExist p do (
				pyScriptPath = p
				return p
			)
			""
		),

		fn pyExec code = (
			try(
				python.execute code
				true
			) catch (
				log ("Python error: " + (getCurrentException() as string))
				false
			)
		),

		fn pyPrelude reloadModule:false = (
			local pyPath = resolvePyPath()
			if pyPath == "" do return ""

			local modDir = escPy (getFilenamePath pyPath)
			local pre = ""
			pre += "import sys\n"
			pre += "mod_dir = '" + modDir + "'\n"
			pre += "if mod_dir not in sys.path:\n"
			pre += "    sys.path.insert(0, mod_dir)\n"
			pre += "import MaxSyncListenerPy\n"
			if reloadModule do (
				pre += "import importlib\n"
				pre += "importlib.reload(MaxSyncListenerPy)\n"
			)
			pre
		),

		fn Start = (
			if isRunning do (
				log "Already running."
				return true
			)

			local pre = pyPrelude reloadModule:true
			if pre == "" then (
				log "MaxSyncListenerPy.py not found."
				return false
			)

			local code = pre
			code += "if not MaxSyncListenerPy.start_listener():\n"
			code += "    raise RuntimeError('start_listener returned False')\n"

			local ok = pyExec code
			isRunning = ok
			if ok then
				log ("Listening on port " + (port as string))
			else
				log "Failed to start Python listener."
			ok
		),

		fn Stop = (
			local pre = pyPrelude reloadModule:false
			if pre != "" do (
				pyExec (pre + "MaxSyncListenerPy.stop_listener()\n")
			)
			isRunning = false
			log "Stopped."
		),

		fn setupFBXExport singleFrame:false = (
			FBXExporterSetParam "Animation" (not singleFrame)
			FBXExporterSetParam "Cameras" false
			FBXExporterSetParam "Lights" false
			FBXExporterSetParam "EmbedTextures" false
			FBXExporterSetParam "SmoothingGroups" true
			FBXExporterSetParam "Skin" true
			FBXExporterSetParam "Shape" true
			FBXExporterSetParam "UpAxis" "Z"
			FBXExporterSetParam "ScaleFactor" 1.0
		),

		fn sendToIClone fbxPath mode:"prop" = (
			local pre = pyPrelude reloadModule:false
			if pre == "" then (
				log "MaxSyncListenerPy.py not found."
				return false
			)

			local pathEsc = escPy fbxPath
			local modeEsc = escPy mode
			local code = pre
			code += "if not MaxSyncListenerPy.send_to_iclone('" + pathEsc + "', '" + modeEsc + "'):\n"
			code += "    raise RuntimeError('send_to_iclone returned False')\n"

			local ok = pyExec code
			if ok then
				log ("Sent to iClone (" + mode + "): " + fbxPath)
			else
				log "Failed to send - is iClone listener running?"
			ok
		),

		fn SendProp = (
			if selection.count == 0 do (
				log "Nothing selected."
				return false
			)

			log ("Exporting " + (selection.count as string) + " object(s)...")
			if doesFileExist tempFile do deleteFile tempFile

			setupFBXExport singleFrame:false
			exportFile tempFile #noPrompt selectedOnly:true using:FBXEXP

			if doesFileExist tempFile then (
				log "Export complete. Sending to iClone..."
				sendToIClone tempFile mode:"prop"
			) else (
				log "Export failed."
				false
			)
		),

		fn SendSingleFrame = (
			if selection.count == 0 do (
				log "Nothing selected."
				return false
			)

			log ("Exporting single frame for " + (selection.count as string) + " object(s)...")
			if doesFileExist tempFile do deleteFile tempFile

			setupFBXExport singleFrame:true
			exportFile tempFile #noPrompt selectedOnly:true using:FBXEXP

			if doesFileExist tempFile then (
				log "Single-frame export complete. Sending to iClone..."
				sendToIClone tempFile mode:"single_frame"
			) else (
				log "Export failed."
				false
			)
		),

		fn Status = (
			local pre = pyPrelude reloadModule:false
			local running = false
			if pre != "" do (
				running = pyExec (pre + "if not MaxSyncListenerPy.status_listener():\n    raise RuntimeError('listener not running')\n")
			)
			isRunning = running
			if running then (
				log ("Running on port " + (port as string))
			) else (
				log "Not running."
			)
			running
		)
	)

	-- Stop existing instance if any
	if MaxSYNC != undefined do (
		try(MaxSYNC.Stop())catch()
	)

	-- Create and start
	MaxSYNC = MaxSYNCStruct()
	MaxSYNC.Start()
)
