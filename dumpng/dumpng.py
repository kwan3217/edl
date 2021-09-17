"""
Decode and dump all of the chunks in a PNG image
"""

import struct

packlen={
    "!I":4,
    "!i":4,
    "!H":2,
    "!h":2,
    "!B":1
}

def decode_struct(payload,fmts):
    pos=0
    for field,fmt in fmts.items():
        if type(fmt) is tuple:
            enm=fmt[1]
            fmt=fmt[0]
        else:
            enm=None
        val=struct.unpack(fmt,payload[pos:pos+packlen[fmt]])[0]
        print(f"   {field}: {val}",end='')
        if enm is not None and val<len(enm):
            print(f" ({enm[val]})")
        else:
            print()

        pos+=packlen[fmt]

def decode_tEXt(payload,fmts):
    k,v=str(payload,encoding='cp437').split("\0")
    print(f"   {k}: {v}")

def ignore_chunk(payload,fmts):
    pass

known_chunks={
    "IHDR":(decode_struct,{"width":"!I",
                           "height":"!I",
                           "Bit depth":"!B",
                           "Color type":("!B",["Grayscale",None,"RGB","Palette","Gray+Alpha",None,"RGB+Alpha"]),
                           "Compression method":("!B",["Flate"]),
                           "Filter_method":("!B",["Adaptive"]),
                           "Interlace metohd":("!B",["None","Adam7"])}),
    "gAMA":(decode_struct,{"Gamma*1e5":"!I"}),
    "sRGB":(decode_struct,{"Rendering intent":("!B",["Perceptual","Relative colorimetric","Saturation","Absolute colorimetric"])}),
    "oFFs":(decode_struct,{"Image position, x axis":"!i",
                           "Image position, y axis":"!i",
                           "Unit specifier":"!B",}),
    "tIME":(decode_struct,{"Year":"!h",
                           "Month":"!B",
                           "Day":"!B",
                           "Hour":"!B",
                           "Minute":"!B",
                           "Second":"!B",}),
    "tEXt":(decode_tEXt,None),
    "IDAT":(ignore_chunk,None),
    "IEND": (decode_struct,{})
}

ignore={"IDAT"}

ref_png_header=bytes([137,80,78,71,13,10,26,10])

def print_bytes(b):
    for i in range(0,len(b),16):
        print(f"{i*16:08x}  ",end='')
        for j in range(16):
            if i+j<len(b):
                print(f"{b[i+j]:02x} ",end='')
            else:
                print("   ",end='')
            if(j==8):
                print(" ",end='')
        print("   ",end='')
        for j in range(16):
            if i+j<len(b):
                if 32<=b[i+j]<=127:
                    print(chr(b[i+j]),end='')
                else:
                    print(".",end='')
            else:
                print(" ",end='')
        print()

def read_chunk(inf):
    clen=struct.unpack("!I",inf.read(4))[0]
    ctype=inf.read(4)
    payload=inf.read(clen)
    crc=struct.unpack("!I",inf.read(4))[0]
    itype=struct.unpack("!I",ctype)[0]
    try:
        stype=str(ctype,encoding='cp437')
    except:
        stype=None
    if stype in ignore:
        return
    print(f"Chunk length: {clen}")
    if stype is None:
        print(f"Chunk type 0x{itype:08x}        length {clen}")
    else:
        print(f"Chunk type 0x{itype:08x} ({stype}) length {clen}")
    if stype in known_chunks:
        return known_chunks[stype][0](payload,known_chunks[stype][1])
    else:
        print("Unknown chunk payload: ")
        print_bytes(payload)

def png_header(inf):
    file_png_header=inf.read(len(ref_png_header))
    print("PNG header: ")
    print_bytes(file_png_header)
    return ref_png_header==file_png_header

def dumpng(infn):
    with open(infn,'rb') as inf:
        png_header(inf)
        while True:
            try:
                read_chunk(inf)
            except Exception:
                break

def main():
    print("file 0 (no version, no gamma)")
    dumpng("/mnt/big/home/chrisj/workspace/edl/Frames_recon/m20_recon_00000.png")
    print("file 1 (no version, yes gamma)")
    dumpng("/mnt/big/home/chrisj/workspace/edl/Frames_recon/m20_recon_00001.png")
    print("file 2 (yes version, no gamma)")
    dumpng("/mnt/big/home/chrisj/workspace/edl/Frames_recon/m20_recon_00002.png")
    print("file 3 (yes version, yes gamma)")
    dumpng("/mnt/big/home/chrisj/workspace/edl/Frames_recon/m20_recon_00003.png")

if __name__=="__main__":
    main()