def main() -> None:
    """Run the DCVC pipeline from the installed project package."""

    from dcvc_compression.dcvc_codec.layered_runner import main as run_main

    run_main()


if __name__ == "__main__":
    main()
