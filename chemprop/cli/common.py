from argparse import ArgumentParser, Namespace
import logging

from chemprop.cli.utils import LookupAction
from chemprop.cli.utils.args import uppercase
from chemprop.featurizers import RxnMode, MoleculeFeaturizerRegistry

logger = logging.getLogger(__name__)


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    data_args = parser.add_argument_group("input data parsing args")
    data_args.add_argument(
        "-s",
        "--smiles-columns",
        nargs="+",
        help="The columns in the input CSV containing SMILES strings. If unspecified, uses the the 0th column.",
    )
    data_args.add_argument(
        "-r",
        "--reaction-columns",
        nargs="+",
        help="The columns in the input CSV containing reactions.",
    )
    # TODO: as we plug the three checkpoint options, see if we can reduce from three option to two or to just one.
    #        similar to how --features-path is/will be implemented
    data_args.add_argument(
        "--checkpoint-dir",
        help="Directory from which to load model checkpoints (walks directory and ensembles all models that are found).",
    )
    data_args.add_argument("--checkpoint-path", help="Path to model checkpoint (:code:`.pt` file).")
    data_args.add_argument(
        "--checkpoint-paths",
        type=list[str],
        help="List of paths to model checkpoints (:code:`.pt` files).",
    )
    # TODO: Is this a prediction only argument?
    parser.add_argument(
        "--checkpoint",
        help="Location of checkpoint(s) to use for ... If the location is a directory, chemprop walks it and ensembles all models that are found. If the location is a path or list of paths to model checkpoints (:code:`.pt` files), only those models will be loaded.",
    )
    data_args.add_argument(
        "--no-cuda", action="store_true", help="Turn off cuda (i.e., use CPU instead of GPU)."
    )
    data_args.add_argument("--gpu", type=int, help="Which GPU to use.")
    data_args.add_argument(
        "--max-data-size", type=int, help="Maximum number of data points to load."
    )
    data_args.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for the parallel data loading (0 means sequential).",
    )
    parser.add_argument("-g", "--n-gpu", type=int, default=1, help="the number of GPU(s) to use")
    data_args.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size.")
    # TODO: The next two arguments aren't in v1. See what they do in v2.
    data_args.add_argument(
        "--no-header-row", action="store_true", help="if there is no header in the input data CSV"
    )

    featurization_args = parser.add_argument_group("featurization args")
    featurization_args.add_argument(
        "--rxn-mode",
        "--reaction-mode",
        type=uppercase,
        default="REAC_DIFF",
        choices=list(RxnMode.keys()),
        help="""Choices for construction of atom and bond features for reactions (case insensitive):
- 'reac_prod': concatenates the reactants feature with the products feature.
- 'reac_diff': concatenates the reactants feature with the difference in features between reactants and products.
- 'prod_diff': concatenates the products feature with the difference in features between reactants and products.
- 'reac_prod_balance': concatenates the reactants feature with the products feature, balances imbalanced reactions.
- 'reac_diff_balance': concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
- 'prod_diff_balance': concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.""",
    )
    featurization_args.add_argument(
        "--keep-h",
        action="store_true",
        help="Whether H are explicitly specified in input (and should be kept this way). This option is intended to be used with the :code:`reaction` or :code:`reaction_solvent` options, and applies only to the reaction part.",
    )
    featurization_args.add_argument(
        "--add-h",
        action="store_true",
        help="Whether RDKit molecules will be constructed with adding the Hs to them. This option is intended to be used with Chemprop's default molecule or multi-molecule encoders, or in :code:`reaction_solvent` mode where it applies to the solvent only.",
    )
    featurization_args.add_argument(
        "--features-generators",
        action=LookupAction(MoleculeFeaturizerRegistry),
        help="Method(s) of generating additional features.",
    )
    featurization_args.add_argument(
        "--features-path",
        type=list[str],  # TODO: why is this a list[str] instead of str?
        help="Path(s) to features to use in FNN (instead of features_generator).",
    )
    featurization_args.add_argument(
        "--phase-features-path",
        help="Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype.",
    )
    featurization_args.add_argument(
        "--no-features-scaling", action="store_true", help="Turn off scaling of features."
    )
    featurization_args.add_argument(
        "--no-atom-descriptor-scaling", action="store_true", help="Turn off atom feature scaling."
    )
    featurization_args.add_argument(
        "--no-bond-descriptor-scaling", action="store_true", help="Turn off bond feature scaling."
    )
    featurization_args.add_argument(
        "--atom-features-path",
        help="Path to the extra atom features. Used as atom features to featurize a given molecule.",
    )
    featurization_args.add_argument(
        "--atom-descriptors-path",
        help="Path to the extra atom descriptors. Used as descriptors and concatenated to the machine learned atomic representation.",
    )
    featurization_args.add_argument(
        "--overwrite-default-atom-features",
        action="store_true",
        help="Overwrites the default atom descriptors with the new ones instead of concatenating them. Can only be used if atom_descriptors are used as a feature.",
    )
    featurization_args.add_argument(
        "--bond-features-path",
        help="Path to the extra bond features. Used as bond features to featurize a given molecule.",
    )
    featurization_args.add_argument(
        "--bond-descriptors-path",
        help="Path to the extra bond descriptors. Used as descriptors and concatenated to the machine learned bond representation.",
    )
    featurization_args.add_argument(
        "--overwrite-default-bond-features",
        action="store_true",
        help="Overwrites the default bond descriptors with the new ones instead of concatenating them. Can only be used if bond_descriptors are used as a feature.",
    )
    # TODO: remove these caching arguments after checking that the v2 code doesn't try to cache.
    # parser.add_argument(
    #     "--no_cache_mol",
    #     action="store_true",
    #     help="Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).",
    # )
    # parser.add_argument(
    #     "--empty_cache",
    #     action="store_true",
    #     help="Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.",
    # )
    # parser.add_argument(
    #     "--cache_cutoff",
    #     type=float,
    #     default=10000,
    #     help="Maximum number of molecules in dataset to allow caching. Below this number, caching is used and data loading is sequential. Above this number, caching is not used and data loading is parallel. Use 'inf' to always cache.",
    # )
    parser.add_argument(
        "--constraints-path",
        help="Path to constraints applied to atomic/bond properties prediction.",
    )

    # TODO: see if we need to add functions from CommonArgs
    return parser


def process_common_args(args: Namespace) -> Namespace:
    return args


def validate_common_args(args):
    pass