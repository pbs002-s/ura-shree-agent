"""
Standalone script to open native OS folder picker dialog in its own process.
This guarantees the dialog runs on the main thread and never freezes asyncio.
"""
import sys
from pathlib import Path

title = sys.argv[1] if len(sys.argv) > 1 else "Select Folder"
initial_dir = sys.argv[2] if len(sys.argv) > 2 else ""

folder = None

# Attempt 1: Tkinter in main thread of this standalone process
try:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    # Bring to foreground on top of browser
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()
    folder = filedialog.askdirectory(
        parent=root,
        title=title,
        initialdir=initial_dir if initial_dir and Path(initial_dir).is_dir() else None,
    )
    root.destroy()
except Exception:
    folder = None

# Attempt 2: PowerShell Shell.Application if Tkinter failed or returned nothing
if not folder and sys.platform == "win32":
    try:
        import subprocess
        # Use PowerShell to invoke native Windows Folder Browser dialog with topmost window
        ps_code = f"""
        Add-Type -AssemblyName System.Windows.Forms
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description = "{title}"
        $dialog.ShowNewFolderButton = $true
        if ("{initial_dir}" -and (Test-Path "{initial_dir}")) {{
            $dialog.SelectedPath = "{initial_dir}"
        }}
        $form = New-Object System.Windows.Forms.Form
        $form.TopMost = $true
        $res = $dialog.ShowDialog($form)
        if ($res -eq [System.Windows.Forms.DialogResult]::OK) {{
            Write-Output $dialog.SelectedPath
        }}
        """
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = res.stdout.strip()
        if out and Path(out).is_dir():
            folder = out
    except Exception:
        pass

if folder:
    print(folder.strip())

