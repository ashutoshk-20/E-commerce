import React, { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar/Navbar";
import Hero from "./components/Hero/Hero";
import Products from "./components/Products/Products";
import Banner from "./components/Banner/Banner";
import Image1 from "./assets/Image1.png";
import Categories from "./components/Categories/Categories";
import ProductDetails from "./components/ProductDetails/ProductDetails";
import { ProductProvider } from "./components/context/ProductContext";
import Image4 from "./assets/Image4.png";
import Blog from "./components/Blog/Blog";
import About from "./components/About/About";

const BannerData = {
  discount: "30% off",
  title: "Fine Smile",
  date: "1 April to 1 May",
  image: Image1,
  title1: "Extreme Bass",
  title2: "Summer Sale",
  title3: "Latest HeadPhones with Top Features",
  bgColor: "#f42c37",
};

const BannerData2 = {
  discount: "30% off",
  title: "Happy Hours",
  date: "1st MAY to 31st MAY",
  image: Image4,
  title1: "Excellent Camera",
  title2: "Summer Sale",
  title3: "Get Latest Phones on Big Discounts !",
  bgColor: "#D23F57",
};

// ✅ Auto-scroll to the top whenever the route changes (Fixes refresh issue)
const ScrollToTop = () => {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]); // Scrolls to top whenever the path changes

  return null;
};

const App = () => {
  return (
    <ProductProvider>
      <Router>
        <ScrollToTop /> {/* ✅ Ensures page starts from the top after refresh */}
        <Navbar />
        <Routes>
          {/* Home Page */}
          <Route
            path="/"
            element={
              <>
                <Hero />
                <Products />
                <Banner data={BannerData} />
                <Categories />
                <Banner data={BannerData2} />
                <Blog />
                <About />
              </>
            }
          />

          {/* Product Details Page */}
          <Route path="/product/:category/:slug" element={<ProductDetails />} />
        </Routes>
      </Router>
    </ProductProvider>
  );
};

export default App;