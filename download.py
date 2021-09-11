"""
Script to download some specific videos from YouTube, in lieu of checking in those videos

Requires youtube-dl
"""

import subprocess
import os

soundtracks={"nJC_pp96wrk":"Elf Meditation",
            "ExfJpuuzxyU":"Jastrian - Sound of Flight"
            }
videos={"UEuOpxOrA_0":"Orion Soars on First Flight Test",     
        "gjglwMPvzVo":"Orion Trial By Fire",    
        "KfMCApWc5xE":"LaunchFlipExplode",
        "1cxkxZ6jlJU":"CFDAndWindTunnels",
        }
   
if True:       
    try:
        os.mkdir("soundtrack")
    except Exception:
        pass
    for hash,obasefn in soundtracks.items():
        cmd=f'youtube-dl -f "bestaudio[ext=m4a]" -o "soundtrack/{obasefn}.%(ext)s" "https://www.youtube.com/watch/?v={hash}"'
        print(cmd)
        subprocess.run(cmd,shell=True)

if True:
    try:
        os.mkdir("fromYoutube")
    except Exception:
        pass
    for hash,obasefn in videos.items():
        cmd=f'youtube-dl -o "fromYoutube/{obasefn}.%(ext)s" "https://www.youtube.com/watch/?v={hash}"'
        print(cmd)
        subprocess.run(cmd,shell=True)


