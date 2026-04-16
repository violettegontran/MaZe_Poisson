#include <stdbool.h>
#ifndef __FORCES_H
#define __FORCES_H

double compute_force_fd(
    int n_grid, int n_p, double h, int num_neigh,
    double *phi, long int *neighbors, double *charges, double *pos, double *forces,
    double (*g)(double, double, double), bool smoothing, double R_c
);
double compute_force_short_range(
    int n_p,
    double *pos,
    double *charges,
    double *forces,
    double R_c,
    double L
);
double compute_tf_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces);
double compute_sc_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces);
double compute_lj_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces);
double compute_coulomb_sr();

#endif