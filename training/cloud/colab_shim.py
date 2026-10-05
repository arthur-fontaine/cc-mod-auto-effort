"""The Colab CLI, asking Google only for the scopes it needs here (see cloud/colab)."""
import sys

import colab_cli.auth as auth

auth.PUBLIC_SCOPES[:] = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
]

from colab_cli.cli import main  # noqa: E402

sys.argv[0] = "colab"
sys.exit(main())
