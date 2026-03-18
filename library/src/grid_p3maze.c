#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "constants.h"
#include "charges.h"
#include "verlet.h"
#include "mp_structs.h"
#include "mpi_base.h"


void  p3maze_grid_init(grid * grid) {
    int n_loc = grid->n_local;
    int n = grid->n;

    long int n2 = n * n;

    maze_multigrid_grid_init_mpi(grid);

    long int size = grid->n_local * n2;
    grid->size = size;

    grid->q = (double *)malloc(size * sizeof(double));
    grid->y = mpi_grid_allocate(n_loc, n);
    grid->phi_p = mpi_grid_allocate(n_loc, n);
    grid->phi_n = mpi_grid_allocate(n_loc, n);

    memset(grid->phi_p, 0, size * sizeof(double));  // phi_p = 0
    memset(grid->phi_n, 0, size * sizeof(double));  // phi_n = 0

    grid->init_field = p3maze_grid_init_field;
    grid->update_field = p3maze_grid_update_field;
    grid->update_charges = p3maze_grid_update_charges;
}

void p3maze_grid_cleanup(grid * grid) {
    free(grid->q);

    mpi_grid_free(grid->y, grid->n);
    mpi_grid_free(grid->phi_p, grid->n);
    mpi_grid_free(grid->phi_n, grid->n);
}

void p3maze_grid_init_field(grid *grid) {
    long int i;

    double *tmp = mpi_grid_allocate(grid->n_local, grid->n);

    double constant = -4 * M_PI / grid->h;
    if ( ! grid->pb_enabled) {
        constant /= grid->eps_s;  // Scale by the dielectric constant if not using PB explicitly
    }

    memset(grid->y, 0, grid->size * sizeof(double));  // y = 0
    vec_copy(grid->phi_n, grid->phi_p, grid->size);  // phi_prev = phi_n
    // phi_n = consant * q
    vec_copy(grid->q, tmp, grid->size);
    dscal(tmp, constant, grid->size);

    if (grid->pb_enabled) {
        conj_grad_pb(
            tmp, grid->y, grid->phi_n, grid->tol, grid->n_local, grid->n,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    } else {
        conj_grad(tmp, grid->y, grid->phi_n, grid->tol, grid->n_local, grid->n);
    }

    mpi_grid_free(tmp, grid->n);
}

int p3maze_grid_update_field(grid *grid) {
    int precond = 1;

    switch (grid->precond_type) {
        case PRECOND_TYPE_NONE:
            precond = 0;
            break;
        default:
            break;
    }

    int res;

    if (grid->pb_enabled) {
           res = verlet_pb_multigrid(
            grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y,
            grid->n_local, grid->n, grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );  
    } else{
        res = verlet_poisson_multigrid(
            grid->tol, grid->h * grid->eps_s, grid->phi_n, grid->phi_p, grid->q, grid->y,
            grid->n_local, grid->n
        );  // grid->h * grid->eps_s to account for the dielectric constant in the poisson equation
    }
    if (precond) {
        fprintf(stderr, "Maze Multigrid with preconditioner not implemented yet.\n");
        exit(1);
    }

    return res;
}   

double p3maze_grid_update_charges(grid *grid, particles *p) {
    double q_tot = update_charges(
        grid->n, p->n_p, grid->h, p->num_neighbors,
        p->pos, p->neighbors, p->charges, grid->q,
        p->charges_spread_func
    );

    //smooth_charges(grid); //to write in charges.c
    return q_tot;
}
