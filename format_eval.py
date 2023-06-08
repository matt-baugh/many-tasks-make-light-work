import argparse

from paper_figures.generate_table import generate_table

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Summaries test set evaluation in a table.")
    parser.add_argument("EXP_PATHS", type=str, nargs='+', help="Paths to experiment file")
    parser.add_argument("--cradl_results", action='store_true', help="Use CRADL results")
    parser.add_argument("--aggregate_folds", action='store_true', help="Aggregate non-ensemble folds")
    parser.add_argument("--metrics", type=str, nargs='+', default=None, help="Metrics to include", required=False)
    parser_args = parser.parse_args()

    results_table = generate_table(parser_args.EXP_PATHS, parser_args.cradl_results, parser_args.aggregate_folds,
                                   parser_args.metrics)
    print(results_table.to_latex(multirow=True, escape=False))
