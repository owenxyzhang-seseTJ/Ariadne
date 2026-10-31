"""Download only the two open Noto fonts required to build the Chinese report."""
from pathlib import Path
import hashlib
import json
import urllib.request


def main():
    target = Path(__file__).resolve().parents[1]/"docs"/"fonts"
    target.mkdir(exist_ok=True)
    base = "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Serif/"
    files = {f"NotoSerifCJKsc-{weight}.otf": base+f"OTF/SimplifiedChinese/NotoSerifCJKsc-{weight}.otf"
             for weight in ["Regular", "Bold"]}
    files["LICENSE"] = base+"LICENSE"
    records=[]
    for name,url in files.items():
        output=target/name
        if output.is_symlink():
            raise ValueError("Refusing to overwrite a font symlink; remove it or use an installed system font")
        urllib.request.urlretrieve(url,output)
        records.append({"file":name,"url":url,"sha256":hashlib.sha256(output.read_bytes()).hexdigest()})
    (target/"provenance.json").write_text(json.dumps(records,indent=2)+"\n")
    print("Downloaded report fonts and their license to",target)


if __name__=="__main__":
    main()
