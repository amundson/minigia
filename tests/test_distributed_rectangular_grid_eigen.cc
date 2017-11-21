#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "distributed_rectangular_grid_eigen.h"
#include "rectangular_grid_domain_eigen_fixture.h"

const double tolerance = 1.0e-12;
int grid_midpoint0 = grid_size0 / 2;

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "construct1", "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        physical_size, physical_offset, grid_shape, is_periodic, 0, grid_size0,
        commxx_sptr);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "construct2", "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        rectangular_grid_domain_eigen, 0, grid_size0, commxx_sptr);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "construct3", "[all]")
{
    std::array<int, 3> padded_shape(
        rectangular_grid_domain_eigen.get_grid_shape());
    padded_shape[2] += 2;
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        rectangular_grid_domain_eigen, 0, grid_size0, padded_shape,
        commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen.get_grid_points().dimension(0) ==
            padded_shape[0]);
    REQUIRE(distributed_rectangular_grid_eigen.get_grid_points().dimension(1) ==
            padded_shape[1]);
    REQUIRE(distributed_rectangular_grid_eigen.get_grid_points().dimension(2) ==
            padded_shape[2]);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "get_const_domain",
                 "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    const Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        rectangular_grid_domain_eigen, 0, grid_size0, commxx_sptr);
    //    BOOST_CHECK_EQUAL(rectangular_grid_domain_eigen,
    //            distributed_rectangular_grid_eigen.get_domain());
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "get_domain", "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        rectangular_grid_domain_eigen, 0, grid_size0, commxx_sptr);
    //    BOOST_CHECK_EQUAL(rectangular_grid_domain_eigen,
    //            distributed_rectangular_grid_eigen.get_domain());
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "periodic_true",
                 "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        physical_size, physical_offset, grid_shape, true, 0, grid_size0,
        commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen.get_domain().is_periodic() ==
            true);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "periodic_false",
                 "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        physical_size, physical_offset, grid_shape, false, 0, grid_size0,
        commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen.get_domain().is_periodic() ==
            false);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "get_lower", "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen1(
        physical_size, physical_offset, grid_shape, false, 0, grid_midpoint0,
        commxx_sptr);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen2(
        physical_size, physical_offset, grid_shape, false, grid_midpoint0,
        grid_size0, commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen1.get_lower() == 0);
    REQUIRE(distributed_rectangular_grid_eigen2.get_lower() == grid_midpoint0);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "get_upper", "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen1(
        physical_size, physical_offset, grid_shape, false, 0, grid_midpoint0,
        commxx_sptr);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen2(
        physical_size, physical_offset, grid_shape, false, grid_midpoint0,
        grid_size0, commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen1.get_upper() == grid_midpoint0);
    REQUIRE(distributed_rectangular_grid_eigen2.get_upper() == grid_size0);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "get_lower_guard",
                 "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen1(
        physical_size, physical_offset, grid_shape, false, 0, grid_midpoint0,
        commxx_sptr);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen2(
        physical_size, physical_offset, grid_shape, false, grid_midpoint0,
        grid_size0, commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen1.get_lower_guard() == 0);
    REQUIRE(distributed_rectangular_grid_eigen2.get_lower_guard() ==
            grid_midpoint0 - 1);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "get_upper_guard",
                 "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen1(
        physical_size, physical_offset, grid_shape, false, 0, grid_midpoint0,
        commxx_sptr);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen2(
        physical_size, physical_offset, grid_shape, false, grid_midpoint0,
        grid_size0, commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen1.get_upper_guard() ==
            grid_midpoint0 + 1);
    REQUIRE(distributed_rectangular_grid_eigen2.get_upper_guard() ==
            grid_size0);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture,
                 "get_lower_guard_periodic", "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen1(
        physical_size, physical_offset, grid_shape, true, 0, grid_midpoint0,
        commxx_sptr);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen2(
        physical_size, physical_offset, grid_shape, true, grid_midpoint0,
        grid_size0, commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen1.get_lower_guard() == -1);
    REQUIRE(distributed_rectangular_grid_eigen2.get_lower_guard() ==
            grid_midpoint0 - 1);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture,
                 "get_upper_guard_periodic", "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen1(
        physical_size, physical_offset, grid_shape, true, 0, grid_midpoint0,
        commxx_sptr);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen2(
        physical_size, physical_offset, grid_shape, true, grid_midpoint0,
        grid_size0, commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen1.get_upper_guard() ==
            grid_midpoint0 + 1);
    REQUIRE(distributed_rectangular_grid_eigen2.get_upper_guard() ==
            grid_size0 + 1);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "get_const_grid_points",
                 "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    const Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        rectangular_grid_domain_eigen, 0, grid_size0, commxx_sptr);
    auto grid_points(distributed_rectangular_grid_eigen.get_grid_points());

    REQUIRE(grid_points.dimension(0) == grid_size0);
    REQUIRE(grid_points.dimension(1) == grid_size1);
    REQUIRE(grid_points.dimension(2) == grid_size2);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "get_grid_points",
                 "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        rectangular_grid_domain_eigen, 0, grid_size0, commxx_sptr);
    auto grid_points(distributed_rectangular_grid_eigen.get_grid_points());

    REQUIRE(grid_points.dimension(0) == grid_size0);
    REQUIRE(grid_points.dimension(1) == grid_size1);
    REQUIRE(grid_points.dimension(2) == grid_size2);
}

TEST_CASE_METHOD(Rectangular_grid_domain_eigen_fixture, "get_set_normalization",
                 "[all]")
{
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_rectangular_grid_eigen distributed_rectangular_grid_eigen(
        rectangular_grid_domain_eigen, 0, grid_size0, commxx_sptr);
    REQUIRE(distributed_rectangular_grid_eigen.get_normalization() ==
            Approx(1.0));
    double new_norm = 123.456;
    distributed_rectangular_grid_eigen.set_normalization(new_norm);
    REQUIRE(distributed_rectangular_grid_eigen.get_normalization() ==
            Approx(new_norm));
}

