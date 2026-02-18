# CloneTools for iClone

Tools to speed up data exchange between 3dsMax and iClone. Comes with a live posing feature.

---

## Tools

### MAXsync

Bidirectional sync system for avatars and props between iClone and 3ds Max.

**iClone → 3ds Max:**
- Send avatar or avatar animation to 3dsmax in one click! It will use "update animation" feature. Use _suffix or _prefix if you have multiple CC characters. Has a fresh import feature to save more time.

- Live posing feature (experimental). You can pose and 3dsmax character will update directly.

**3ds Max → iClone:**
- Send selected meshes to iClone (single frame or animated)
- Send pose to iClone
---

### CloneFBX

This is an FBX exporter with option to remove mesh and more hidden parameters that cannot be found within iClone's standard FBX exporter. It just reveals hidden parameters in the API.

**Options:**
- Export / zero out root motion
- Remove all mesh (skeleton export only)
- T-pose on motion first frame
- Remove bone root
- Up axis: Z-up, X-up, Y-up, UE4 bone axis

Registers under `Plugins > CloneTools > CloneFBX` in iClone.

---

## Installation

### 3ds Max (MAXsync)

1. Drag `MaxSYNC/MaxSyncMacros.ms` into 3ds Max.
2. This auto-detects its location, copies both `.ms` files to your user scripts folder, and installs a startup loader so MAXsync initializes automatically on next launch.
3. Toolbar buttons and menu items register under **Customize User Interface > Category: CloneTools**.

### iClone (MAXsync + CloneFBX)

Load the plugin scripts via **Menu > Script > Load Python**:
- `MaxSYNC/MaxSYNC.py`
- `CloneFBX/CloneFBX.py`

Both register their UI under the **Plugins** menu in iClone.
---

## How to use

**Sync an avatar from iClone to 3ds Max:**
1. Select your avatar in iClone.
2. In the MAXsync panel, choose export options and click **SYNC Avatar**.
3. The avatar FBX is sent to 3ds Max and imported with your chosen mode.

**Live posing:**
1. Select bones/objects in iClone.
2. Click **Start Live Stream** in the MAXsync panel.
3. Transforms stream continuously to matching nodes in 3ds Max.

**Send a prop from 3ds Max to iClone:**
1. Select objects in 3ds Max.
2. Click **Send to iClone** (MAXsync toolbar).
3. The prop is exported and loaded into iClone automatically.

---

## License

MIT License
