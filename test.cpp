#include <igl/cotmatrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <string>
#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <stdio.h>
#include <igl/opengl/glfw/Viewer.h>
using namespace std;




int main(int argc, char* argv[])
{
    Eigen::MatrixXd V,U;
    Eigen::MatrixXi F;
    string filename(argv[1]);
    fprintf(stdout, "Filename : %s\n", filename.c_str());
    igl::readOFF(filename,V,F);
    fprintf(stdout, "V : rows %ld cols %ld\nF : rows %ld cols %ld\n", V.rows(), V.cols(), F.rows(), F.cols());

    Eigen::SparseMatrix<double> Adj(V.rows(), V.rows());
    
    for(auto i : F.rowwise())
    {
        //cout << i[0] << i[1] << i[2];
        Adj.coeffRef(i[0], i[1]) = 1;
        Adj.coeffRef(i[1], i[0]) = 1;
        Adj.coeffRef(i[1], i[2]) = 1;
        Adj.coeffRef(i[2], i[1]) = 1;
        Adj.coeffRef(i[0], i[2]) = 1;
        Adj.coeffRef(i[2], i[0]) = 1;
    }

    //uniform laplacian
    for(int i = 0; i < V.rows(); i++)
    {
        Adj.coeffRef(i, i) = -Adj.row(i).sum();
    }

    //iterate throuth the adj matrix
    for(int k = 0; k < Adj.outerSize(); ++k)
    {
        {
            for(Eigen::SparseMatrix<double>::InnerIterator it(Adj, k); it; ++it)
            {
                fprintf(stdout, "adj : row %ld col %ld value %f\n", it.row(), it.col(), it.value());
            }
        }
    }

    //find the boundary
    Eigen::VectorXi bnd;
    igl::boundary_loop(F,bnd);

    fprintf(stdout, "bnd : row %ld col %ld \n", bnd.rows(), bnd.cols());


    //map vertices to circle
    Eigen::MatrixXd uv_bnd;
    igl::map_vertices_to_circle(V, bnd, uv_bnd);

    fprintf(stdout, "uv_bnd row %ld col %ld \n", uv_bnd.rows(), uv_bnd.cols());
    
    Eigen::MatrixXd uv0 = Eigen::ArrayXd::Zero(V.rows());
    Eigen::MatrixXd uv1 = Eigen::ArrayXd::Zero(V.rows());
    
    for(int i = 0; i < bnd.rows(); ++i)
    {
        uv0(bnd(i)) = uv_bnd(i, 0);
        uv1(bnd(i)) = uv_bnd(i, 1);   
        fprintf(stdout, "bnd %d : (%f, %f) \n", i, uv_bnd(i, 0), uv_bnd(i, 1));
    }

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
    solver.compute(Adj);
    if(solver.info() != Eigen::Success) {
        // decomposition failed
        fprintf(stdout, "decom fail\n");
        return 0;
    }
    Eigen::MatrixXd x; x = solver.solve(uv0);
    Eigen::MatrixXd y; y = solver.solve(uv1);
    if(solver.info()!=Eigen::Success) {
        // solving failed
        fprintf(stdout, "solver fail\n");
        fprintf(stdout, "using least square incerse\n");
        Eigen::MatrixXd mtx = Adj.transpose()*Adj;
        mtx = mtx.inverse()*Adj.transpose();
        x = mtx*uv0;
        y = mtx*uv1;
 
    }


    Eigen::MatrixXd V_uv(x.rows(), x.cols() + y.cols());
    V_uv << x, y;
    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().set_uv(V_uv);
    viewer.callback_key_pressed = 
    [&V,&V_uv,&F](igl::opengl::glfw::Viewer& viewer, unsigned int key, int /*mod*/)
    {
      if(key == '3' || key == '2')
      {
        // Plot the 3D mesh or 2D UV coordinates
        viewer.data().set_vertices(key=='3'?V:V_uv);
        viewer.data().compute_normals();
        viewer.core().align_camera_center(key=='3'?V:V_uv,F);
        // key press was used
        return true;
      }
      // key press not used
      return false;
    };
    viewer.launch();
    FILE* fp;
    fp =  fopen(argv[2], "w");    

    for(int i = 0; i < uv0.rows(); ++i)
    {
        fprintf(fp, "%f %f\n", x(i), y(i));
    }
    fclose(fp);


    


    return 0;
}

