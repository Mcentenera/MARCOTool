def print_results(results):
    def print_section(title):
        print(f"\n{'=' * 60}\n{title:^60}\n{'=' * 60}")

    print_section("TELESCOPE")
    keys_tel = [k for k in results if any(s in k for s in ["OTA", "Seeing", "Input", "Module", "F-number", "Focal length"])]
    for k in keys_tel:
        print(f"{k:<45}: {results[k]}")

    print_section("PHOTONIC LANTERN")
    keys_pl = [k for k in results if any(s in k for s in ["Expected", "Modes", "Fibers", "Required output", "Selected", "Super-PL"])]
    for k in keys_pl:
        print(f"{k:<45}: {results[k]}")

    print_section("SPECTROGRAPH")
    keys_instr = [k for k in results if any(s in k for s in ["Beam diameter", "Spectrograph", "Estimated cost", "Magnification", "Resolution"])]
    for k in keys_instr:
        print(f"{k:<45}: {results[k]}")

    print_section("AUXILIAR PARAMETERS")
    other_keys = [k for k in results if k not in keys_tel + keys_pl + keys_instr and k != ""]
    for k in other_keys:
        print(f"{k:<45}: {results[k]}")

