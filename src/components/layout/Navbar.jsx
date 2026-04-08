import { Link, NavLink } from "react-router-dom";

function Navbar() {
  return (
    <header className="site-header">
      <div className="site-header__inner">
        <Link className="brand" to="/">
          <span className="brand__mark">PlantCare</span>
        </Link>

        <nav className="site-nav" aria-label="Primary">
          <NavLink
            to="/"
            className={({ isActive }) =>
              isActive ? "site-nav__link site-nav__link--active" : "site-nav__link"
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/detect"
            className={({ isActive }) =>
              isActive ? "site-nav__link site-nav__link--active" : "site-nav__link"
            }
          >
            Detection
          </NavLink>
        </nav>
      </div>
    </header>
  );
}

export default Navbar;
