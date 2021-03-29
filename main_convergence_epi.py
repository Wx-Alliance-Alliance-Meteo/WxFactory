#!/usr/bin/env python3
from convergence import ConvergenceTools
from timeIntegrators import Epi

def main(args) -> int:
   
    CA = ConvergenceTools.ConvergenceAnalyzer(args)
    output_filepath = "output/epi-results.json"
    Nts = [1, 4, 8, 16, 32, 64, 128]
    Nt_reference = 4096
    integrators = [
        {
            "name" : "epi2",
            "object" : Epi(2, CA.rhs, 1e-3, 3, init_substeps=10)
        },
        {
            "name" : "epi3",
            "object" : Epi(3, CA.rhs, CA.params.tolerance, CA.params.krylov_size, init_substeps=10)
        },
        {
            "name" : "epi4",
            "object" : Epi(4, CA.rhs, CA.params.tolerance, CA.params.krylov_size, init_substeps=10)
        },
        {
            "name" : "epi5",
            "object" : Epi(5, CA.rhs, CA.params.tolerance, CA.params.krylov_size, init_substeps=10)
        },
        {
            "name" : "epi6",
            "object" : Epi(6, CA.rhs, CA.params.tolerance, CA.params.krylov_size, init_substeps=10)
        }
    ]

    CA.analyze(output_filepath, integrators, Nts, Nt_reference=Nt_reference)

if __name__ == '__main__':

   import argparse

   parser = argparse.ArgumentParser(description='Solve NWP problems with GEF!')
   parser.add_argument('--profile', action='store_true', help='Produce an execution profile when running')
   parser.add_argument('config', type=str, help='File that contains simulation parameters')
   args = parser.parse_args()

   main(args)