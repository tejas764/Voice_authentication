import argparse
import os

from auth_service import VoiceAuthService


def file_choices(choices, filename):
    ext = os.path.splitext(filename)[1][1:].lower()
    if ext not in choices:
        raise argparse.ArgumentTypeError(f"File extension must be one of {choices}")
    return filename


def build_parser():
    parser = argparse.ArgumentParser(description="Voice authentication using CNN embeddings.")
    parser.add_argument(
        "-t",
        "--task",
        choices=["enroll", "recognize"],
        required=True,
        help='Task to do: "enroll" or "recognize"',
    )
    parser.add_argument(
        "-n",
        "--name",
        required=False,
        help="Name of the speaker for enrollment",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        type=lambda fn: file_choices(("csv", "wav", "flac"), fn),
        help="Audio file path (.wav/.flac) or enrollment CSV file",
    )
    return parser


def main():
    args = build_parser().parse_args()
    service = VoiceAuthService()

    extension = os.path.splitext(args.file)[1][1:].lower()

    if args.task == "enroll":
        if extension == "csv":
            count = service.enroll_csv(args.file)
            print(f"Successfully enrolled {count} user(s).")
            return
        if not args.name:
            raise ValueError("For single-file enrollment, provide -n/--name.")
        out_file = service.enroll(args.name, args.file)
        print(f"Successfully enrolled '{args.name}' -> {out_file}")
        return

    if extension == "csv":
        raise ValueError("Recognition expects an audio file (.wav/.flac), not CSV.")

    result = service.recognize(args.file)
    if result["accepted"]:
        print(f"Recognized: {result['speaker']}")
    else:
        print("Could not identify the user. Try enrolling with a clearer sample.")
    print(
        f"Score: {result['score']:.4f} ({result['metric']}), "
        f"Threshold: {result['threshold']:.4f}"
    )


if __name__ == "__main__":
    main()
