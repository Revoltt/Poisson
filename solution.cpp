#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define R2(x,y) ((x) * (x) + (y) * (y))

#define hx(i)  (XNodes[i + 1] - XNodes[i])
#define hy(j)  (YNodes[j + 1] - YNodes[j])

const int NX = 7, NY= 7;
const double q = 1.5;
int PX, PY; // amount of processors on each dimension
double *XNodes, *YNodes; // lists for x and y coordinates in mesh nodes
double *p, *pnew; // solution
double *r; // residual matrice
double *l_r, *l_g; // laplace operator of residual matrice
double *g; // additional matrice
double *err; // error vect

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
    int kx = NX % PX;
    *startx = (NX - 1) / PX * (num % PX) + 1;    
    *finishx = (NX - 1) / PX * (num % PX + 1);
    if (num % PX > PX - kx)
    {
        //printf("%d\n", num - PX + kx);
        
        *startx += num % PX - PX + kx - 1;
        *finishx += num % PX - PX + kx;
    }
    
    int ky = NY % PY;
    *starty = (NY - 1) / PY * (num / PY) + 1;    
    *finishy = (NY - 1) / PY * (num / PY + 1);
    if (num / PY > PY - ky)
    {
        //printf("%d\n", num - PY + ky);
        
        *starty += num / PY - PY + ky - 1;
        *finishy += num / PY - PY + ky;
    }
    
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

void printM(double * p)
{
    for (int j = 0; j <= NY; j++)
    {
        for (int i = 0; i <= NX; i++)
        {
            printf("%f ", p[j * (NX + 1) + i]);
        }
        printf("\n");
    }
    printf("\n");
}
//==============================================================================

int main(int argc, char *argv[])
{
    int procNum = 1;
    int rank;
    
    if (procGrid(procNum) != -1)
        printf("%d %d\n", PX, PY);
    
    XNodes = (double *)malloc((NX + 1) * sizeof(double));
    YNodes = (double *)malloc((NY + 1) * sizeof(double));
    
    genMesh(NX, NY);
    // solution we want to compute 
    p = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    pnew = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    memset(p, 0, (NX + 1) * (NY + 1) * sizeof(double));
    initP();
    
    // residual matrice
    r = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    l_r = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    // additional matrice
    g = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    l_g = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    err = (double *)malloc((NX + 1) * (NY + 1) * sizeof(double));
    memset(r, 0, (NX + 1) * (NY + 1) * sizeof(double));
    int startx, starty, finishx, finishy;
    
    int iteration = 0;
    while (true) // main iteration loop
    {
        double tau = 0, part_tau = 0;
        double alpha = 0, part_alpha = 0;
        double sumErr = 0;
        
        // for each processor
        rank = 0;
        procArea(rank, &startx, &finishx, &starty, &finishy);
        //printf("%d: %d %d %d %d\n", rank, startx, finishx, starty, finishy);
        
        if (iteration == 0)
        {
            // computing r. r on iteration zero can be counted without sending information about p between processes
            // on iteration zero g is equal to r
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    r[(NX + 1) * j + i] = Laplace(p, i, j) - F(XNodes[i], YNodes[j]);
                    g[(NX + 1) * j + i] = r[(NX + 1) * j + i];
                }
            }
                
            // count laplacian of r
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    l_r[(NX + 1) * j + i] = Laplace(r, i, j);
                }
            }
            part_tau = scalarProduct(r, r, startx, finishx, starty, finishy) / scalarProduct(l_r, r, startx, finishx, starty, finishy);
            tau += part_tau;
            // TODO broadcast tau
            // count p(k + 1)
//            printM(p);
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    double temp = p[(NX + 1) * j + i] - tau * r[(NX + 1) * j + i];
                    err[(NX + 1) * j + i] = fabs(temp - p[(NX + 1) * j + i]);
                    p[(NX + 1) * j + i] = temp;
                }
            }
//            printM(r);
//            printM(p);
//            printf("%f\n", tau);
        } else
        {
            // count r(k)
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    r[(NX + 1) * j + i] = Laplace(p, i, j) - F(XNodes[i], YNodes[j]);
                }
            }
          
            // count laplacian of r
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    l_r[(NX + 1) * j + i] = Laplace(r, i, j);
                }
            }
                
            // count laplacian of g
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    l_g[(NX + 1) * j + i] = Laplace(g, i, j);
                }
            }
                
            // count alpha
            part_alpha = scalarProduct(l_r, g, startx, finishx, starty, finishy) / scalarProduct(l_g, g, startx, finishx, starty, finishy);
            // TODO gather part_alpha
            alpha += part_alpha;
            // TODO broadcast alpha
            
            // count g
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    g[(NX + 1) * j + i] = r[(NX + 1) * j + i] - alpha * g[(NX + 1) * j + i];
                }
            }
                
            // count laplacian of g
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    l_g[(NX + 1) * j + i] = Laplace(g, i, j);
                }
            }
                
            part_tau = scalarProduct(r, g, startx, finishx, starty, finishy) / scalarProduct(l_g, g, startx, finishx, starty, finishy);
            // TODO gather part_tau
            tau += part_tau;
            // TODO broadcast tau
            // count p(k + 1)
//            printM(p);
            for (int j = starty; j <= finishy; j++)
            {
                for (int i = startx; i <= finishx; i++)
                {
                    double temp = p[(NX + 1) * j + i] - tau * g[(NX + 1) * j + i];
                    err[(NX + 1) * j + i] = fabs(temp - p[(NX + 1) * j + i]);
                    p[(NX + 1) * j + i] = temp;
                }
            }
//            printM(r);
//            printM(p);
//            printM(g);
//            printf("%f\n", tau);
//            printf("%f\n", alpha);
        }
        
        iteration++;
        // reduce sumErr
        sumErr += scalarProduct(err, err, startx, finishx, starty, finishy);
        
        printf("%f\n", sumErr);
        //if (sumErr <= eps * eps)
        //    break;
        if (iteration > 10)
            break;
    }
    return 0;
}