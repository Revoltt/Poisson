#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define R2(x,y) ((x) * (x) + (y) * (y))

#define hx(i)  (XNodes[i + 1] - XNodes[i])
#define hy(j)  (YNodes[j + 1] - YNodes[j])

int NX = 7, NY= 7;
const double q = 1.5;
int PX, PY; // amount of processors on each dimension
double *XNodes, *YNodes; // lists for x and y coordinates in mesh nodes
double *p, *pnew; // solution
double *r; // residual matrice
double *l_r, *l_g; // laplace operator of residual matrice
double *g; // additional matrice
double *err; // error vect
int size, rank; // processor sizeand processor rank
    
const double A1 = 0.0, A2 = 1.0, B1 = 0.0, B2 = 1.0;
const double eps = 1e-4;
//==============================Utility functions===============================
double scalarProduct(double *a, double *b, int startx, int finishx, int starty, int finishy)
{
    double result = 0.0;
    for (int j = starty; j <= finishy; j++)
    {
        for (int i = startx; i <= finishx; i++)
        {
            result += a[j * (NX + 1) + i] * b[j * (NX + 1) + i] * \
                    0.5 * (hx(i) + hx(i - 1)) * 0.5 * (hy(j) + hy(j - 1));
        }
    }
    return result;
}
//==============================================================================
//=======================Mesh element values====================================
double f(double x) // function for nonuniform mesh element values
{
    return (pow(1.0 + x, q) - 1.0) / (pow(2.0, q) - 1.0);
}

int genMesh(int NX, int NY) // gets x and y values of mesh nodes
{
    int i;

    for(i = 0; i <= NX; i++)
        XNodes[i] = A2 * f(1.0 * i / NX) + A1 * (1.0 - f(1.0 * i / NX));
	
    for(i = 0; i <= NY; i++)
        YNodes[i] = B2 * f(1.0 * i / NY) + B1 * (1.0 - f(1.0 * i / NY));
    return 0;
}
//==============================================================================
//========================MPI processor grid====================================
int isPower(int num) // check is num is power of 2, returns log
{
    unsigned int temp;
    int p;
    
    if(num <= 0)
        return -1;
        
    temp = num; p = 0;
    while(temp % 2 == 0)
    {
        ++p;
        temp = temp >> 1;
    }
    if((temp >> 1) != 0)
        return -1;
    else
        return p;
}

int procGrid(int procNum)
{
    int p = isPower(procNum);
    if (p == -1)
    {
        printf("Processor number is not power of 2\n");
        return -1; 
    }
    if (p % 2 == 0)
    {
        PX = 1 << p/2;
        PY = 1 << p/2;
    } else
    {
        PX = 1 << p/2;
        PY = 1 << (p/2 + 1);
    }
    return 0;
}

int procArea(int num, int *startx, int *finishx, int *starty, int *finishy) 
// counts rectangle in which processor num is responsible for computations
{
    int rx = (NX - 1) % PX; // amount of processors with higher amount of points
    int kx = (NX - 1) / PX; // minimal amount of points for processor
    if (num % PX < rx) // processor will have more points 
    {
        kx += 1;
        *startx = num % PX * kx + 1;
    } else
    {
        *startx = (NX - 1) - kx * (PX - num % PX) + 1;
    }
    *finishx = *startx + kx - 1;    
    
    int ry = (NY - 1) % PY; // amount of processors with higher amount of points
    int ky = (NY - 1) / PY; // minimal amount of points for processor
    if (num / PX < ry) // processor will have more points 
    {
        ky += 1;
        *starty = num / PX * ky + 1;
    } else
    {
        *starty = (NY - 1) - ky * (PY - num / PX) + 1;
    }
    *finishy = *starty + ky - 1;    
    
    return 0;
}
//==============================================================================
//===========================Equation functions=================================
double rSolution(double x,double y) // precise solution is equal to boundary equations 
{
    return R2(1 - x * x, 1 - y * y);
}

double fi(double x, double y) // boundary function phi
{
    return rSolution(x,y);
}

double F(double x, double y) // function F of puasson equation
{
    return 4 * (2 - 3 * R2(x, y));
}

// 5-point Laplace operator for function p in point (i, j)
#define Laplace(P, i, j)\
((-(P[(NX + 1) * (j) + i + 1] - P[(NX + 1) * (j) + i]) / hx(i) + \
(P[(NX + 1) * (j) + i] - P[(NX + 1) * (j) + i - 1]) / hx(i - 1)) / (0.5 * (hx(i) + hx(i - 1))) + \
(-(P[(NX + 1) * (j + 1) + i] - P[(NX + 1) * (j) + i]) / hy(j) + \
(P[(NX + 1) * (j) + i] - P[(NX + 1) * (j - 1) + i]) / hy(j - 1)) / (0.5 * (hy(j) + hy(j - 1))))

void initP() // initialization of p
{
    for(int i = 0; i <= NX; i++)
    {
        p[i] = fi(XNodes[i],B1);
        p[NX * (NY + 1) + i] = fi(XNodes[i], B2);
    }
    
    for(int j = 0; j <= NY; j++)
    {
        p[(NX + 1) * j] = fi(A1, YNodes[j]);
        p[(NX + 1) * j +  NX] = fi(A2, YNodes[j]);
    }     
}

void printM(double * p, int startx, int finishx, int starty, int finishy) // print matrice
{
    for (int j = starty - 1; j <= finishy + 1; j++)
    {
        for (int i = startx - 1; i <= finishx + 1; i++)
        {
            printf("%f ", p[j * (NX + 1) + i]);
        }
        printf("\n");
    }
    printf("\n");
}
//==============================================================================
//============================MPI send-receive all==============================
MPI_Request *reqs;
MPI_Request *reqr;

void sendAll(double *matrice, int xstart, int xfinish, int ystart, int yfinish)
{
    reqs = (MPI_Request *)malloc(4 * sizeof(MPI_Request));

    double *left = (double *)malloc((yfinish - ystart + 1) * sizeof(double));
    double *right = (double *)malloc((yfinish - ystart + 1) * sizeof(double));
    double *top = (double *)malloc((xfinish - xstart + 1) * sizeof(double));
    double *bottom = (double *)malloc((xfinish - xstart + 1) * sizeof(double));

    //printM(matrice, xstart, xfinish, ystart, yfinish);
    if (rank % PX != 0) // left
    {
        for (int i = ystart; i <= yfinish; i++)
        {
            left[i - ystart] = matrice[i * (NX + 1) + xstart];
        }
    }

    if (rank % PX != PX - 1) // right
    {
        for (int i = ystart; i <= yfinish; i++)
        {
             right[i - ystart] = matrice[i * (NX + 1) + xfinish];
        }
    }
    
    if (rank / PX != 0) // top
    {
        for (int i = xstart; i <= xfinish; i++)
        {
            top[i - xstart] = matrice[ystart * (NX + 1) + i];
        }
    }
    if (rank / PX != PY - 1) // bottom
    {
        for (int i = xstart; i <= xfinish; i++)
        {
            bottom[i - xstart] = matrice[yfinish * (NX + 1) + i];
        }
    }

    if (rank % PX != 0)
        MPI_Isend(left, yfinish - ystart + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[0]);
    if (rank % PX != PX - 1)
        MPI_Isend(right, yfinish - ystart + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[1]);
    if (rank / PX != 0)
        MPI_Isend(top, xfinish - xstart + 1, MPI_DOUBLE, rank - PX, 0, MPI_COMM_WORLD, &reqs[2]);
    if (rank / PX != PY - 1)
        MPI_Isend(bottom, xfinish - xstart + 1, MPI_DOUBLE, rank + PX, 0, MPI_COMM_WORLD, &reqs[3]);
    free(reqs);
}

void receiveAll(double *matrice, int xstart, int xfinish, int ystart, int yfinish)
{
    reqr = (MPI_Request *)malloc(4 * sizeof(MPI_Request));

    double *left = (double *)malloc((yfinish - ystart + 1) * sizeof(double));
    double *right = (double *)malloc((yfinish - ystart + 1) * sizeof(double));
    double *top = (double *)malloc((xfinish - xstart + 1) * sizeof(double));
    double *bottom = (double *)malloc((xfinish - xstart + 1) * sizeof(double));
    
    if (rank % PX != 0) 
    {
        MPI_Irecv(left, yfinish - ystart + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqr[0]);
        MPI_Wait(&reqr[0], MPI_STATUS_IGNORE);
    }
    if (rank % PX != PX - 1)
    {
        MPI_Irecv(right, yfinish - ystart + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqr[1]);
        MPI_Wait(&reqr[1], MPI_STATUS_IGNORE);
    }
    if (rank / PX != 0)
    {
        MPI_Irecv(top, xfinish - xstart + 1, MPI_DOUBLE, rank - PX, 0, MPI_COMM_WORLD, &reqr[2]);
        MPI_Wait(&reqr[2], MPI_STATUS_IGNORE);
    }
    if (rank / PX != PY - 1)
    {
        MPI_Irecv(bottom, xfinish - xstart + 1, MPI_DOUBLE, rank + PX, 0, MPI_COMM_WORLD, &reqr[3]);
        MPI_Wait(&reqr[3], MPI_STATUS_IGNORE);
    }

    if (rank % PX != 0) // left
    {
        for (int i = ystart; i <= yfinish; i++)
        {
            matrice[i * (NX + 1) + xstart - 1] = left[i - ystart];
        }
    }

    if (rank % PX != PX - 1) // right
    {
        for (int i = ystart; i <= yfinish; i++)
        {
             matrice[i * (NX + 1) + xfinish + 1] = right[i - ystart];
        }
    }
    
    if (rank / PX != 0) // top
    {
        for (int i = xstart; i <= xfinish; i++)
        {
            matrice[(ystart - 1) * (NX + 1) + i] = top[i - xstart];
        }
    }
    if (rank / PX != PY - 1) // bottom
    {
        for (int i = xstart; i <= xfinish; i++)
        {
            matrice[(yfinish + 1) * (NX + 1) + i] = bottom[i - xstart];
        }
    }
    free(reqr);
}
//==============================================================================
int main(int argc, char *argv[])
{
    //int procNum = 1;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    NX = atoi(argv[1]);
    NY = atoi(argv[2]);
    procGrid(size);
    //if (procGrid(size) != -1)
    //    printf("%d %d\n", PX, PY);
    
    XNodes = (double *)malloc((NX + 1) * sizeof(double));
    YNodes = (double *)malloc((NY + 1) * sizeof(double));
    
    genMesh(NX, NY);
    // solution we want to compute 
    p = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    memset(p, 0, (NX + 1) * (NY + 1) * sizeof(double));
    initP();
    
    // residual matrice
    r = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    // laplace of residual matrice
    l_r = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    // additional matrice
    g = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    // laplace of additional matrice
    l_g = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    // error matrice
    err = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    
    memset(r, 0, (NX + 1) * (NY + 1) * sizeof(double));
    int startx, starty, finishx, finishy;
    
    procArea(rank, &startx, &finishx, &starty, &finishy);
    //printf("%d: %d %d %d %d\n", rank, startx, finishx, starty, finishy);
    
    int iteration = 0;
    while (true) // main iteration loop
    {
        double tau = 0, part_tau_top = 0, part_tau_bottom, tau_top, tau_bottom;
        double alpha = 0, part_alpha_top = 0, part_alpha_bottom, alpha_top, alpha_bottom;
        double sumErr = 0;
        
        sendAll(p, startx, finishx, starty, finishy);
        receiveAll(p, startx, finishx, starty, finishy);
        //MPI_Barrier(MPI_COMM_WORLD);
    
        // computing residual matrice and g if iteration is zero
        for (int j = starty; j <= finishy; j++)
        {
            for (int i = startx; i <= finishx; i++)
            {
                r[(NX + 1) * j + i] = Laplace(p, i, j) - F(XNodes[i], YNodes[j]);
                if (iteration == 0)
                    g[(NX + 1) * j + i] = r[(NX + 1) * j + i];
            }
        }
	sendAll(r, startx, finishx, starty, finishy); 
        receiveAll(r, startx, finishx, starty, finishy);
        //printM(r, startx, finishx, starty, finishy);
        //MPI_Barrier(MPI_COMM_WORLD);
    
        // count laplacian of r
        for (int j = starty; j <= finishy; j++)
        {
            for (int i = startx; i <= finishx; i++)
            {
                l_r[(NX + 1) * j + i] = Laplace(r, i, j);
                if (iteration == 0)
                    l_g[(NX + 1) * j + i] = l_r[(NX + 1) * j + i];
            }
        }
        //printM(l_r, startx, finishx, starty, finishy);
        
        if (iteration == 0)
        {
            part_tau_top = scalarProduct(r, r, startx, finishx, starty, finishy);
            part_tau_bottom = scalarProduct(l_r, r, startx, finishx, starty, finishy);
            MPI_Reduce(&part_tau_top, &tau_top, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // gather part_tau
            MPI_Bcast(&tau_top, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // broadcast tau
            MPI_Reduce(&part_tau_bottom, &tau_bottom, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // gather part_tau
            MPI_Bcast(&tau_bottom, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // broadcast tau
            tau = tau_top / tau_bottom;
            // count p(k + 1)
            //printM(r, startx, finishx, starty, finishy);
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    double temp = p[(NX + 1) * j + i] - tau * r[(NX + 1) * j + i];
                    err[(NX + 1) * j + i] = fabs(temp - p[(NX + 1) * j + i]);
                    p[(NX + 1) * j + i] = temp;
                }
            }
//            printM(r, startx, finishx, starty, finishy);
//            printM(p, startx, finishx, starty, finishy);
//            printf("%f\n", tau);
        } else
        {        
            // count alpha
            part_alpha_top = scalarProduct(l_r, g, startx, finishx, starty, finishy);
            part_alpha_bottom =  scalarProduct(l_g, g, startx, finishx, starty, finishy);
            MPI_Reduce(&part_alpha_top, &alpha_top, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // gather part_alpha
            MPI_Bcast(&alpha_top, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // broadcast alpha
            MPI_Reduce(&part_alpha_bottom, &alpha_bottom, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // gather part_alpha
            MPI_Bcast(&alpha_bottom, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // broadcast alpha
            alpha = alpha_top / alpha_bottom;

            // count g
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    g[(NX + 1) * j + i] = r[(NX + 1) * j + i] - alpha * g[(NX + 1) * j + i];
                }
            }
                
            sendAll(g, startx, finishx, starty, finishy);
            receiveAll(g, startx, finishx, starty, finishy);
            //MPI_Barrier(MPI_COMM_WORLD);
    
            // count laplacian of g
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    l_g[(NX + 1) * j + i] = Laplace(g, i, j);
                }
            }
                
            part_tau_top = scalarProduct(r, g, startx, finishx, starty, finishy);
            part_tau_bottom = scalarProduct(l_g, g, startx, finishx, starty, finishy);
            MPI_Reduce(&part_tau_top, &tau_top, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // gather part_tau
            MPI_Bcast(&tau_top, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // broadcast tau
            MPI_Reduce(&part_tau_bottom, &tau_bottom, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // gather part_tau
            MPI_Bcast(&tau_bottom, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // broadcast tau
            tau = tau_top / tau_bottom;
            // count p(k + 1)
            //printM(p, startx, finishx, starty, finishy);
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    double temp = p[(NX + 1) * j + i] - tau * g[(NX + 1) * j + i];
                    err[(NX + 1) * j + i] = fabs(temp - p[(NX + 1) * j + i]);
                    p[(NX + 1) * j + i] = temp;
                }
            }
//            printM(r, startx, finishx, starty, finishy);
//            printM(p, startx, finishx, starty, finishy);
//            printM(g, startx, finishx, starty, finishy);
//            printf("%f\n", tau);
//            printf("%f\n", alpha);
        }
//        if (iteration == 0) printM(p, startx, finishx, starty, finishy);
        iteration++;
        double part_err = scalarProduct(err, err, startx, finishx, starty, finishy); // reduce sumErr
        MPI_Reduce(&part_err, &sumErr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // broadcast sumErr
        MPI_Bcast(&sumErr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
            printf("%.10f\n", sumErr);
        if (sumErr <= eps * eps)
            break;
        //if (iteration > 5)
        //    break;
    }
    free(XNodes);
    free(YNodes);
    free(p);
    free(r);
    free(l_r);
    free(g);
    free(l_g);
    free(err);
    
    MPI_Finalize();
    
    return 0;
}