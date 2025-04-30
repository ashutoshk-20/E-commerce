import React from "react";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import Image1 from "../../assets/Image1.png";
import Image2 from "../../assets/Image2.png";
import Image3 from "../../assets/Image3.png";
import Image4 from "../../assets/Image4.png";
import Button from "../Shared/Button";
import SearchBar from "../SearchBar/SearchBar";

const HeroData = [
    {
        id: 1,
        img: Image1,
        subtitle: "Beats Solo",
        title: "Wireless",
        title2: "Headphones",
    },
    {
        id: 2,
        img: Image2,
        subtitle: "Beats Solo",
        title: "Wireless",
        title2: "VirtualReality",
    },
    {
        id: 3,
        img: Image3,
        subtitle: "Beats Solo",
        title: "Branded",
        title2: "Laptops",
    },
    {
        id: 4,
        img: Image4,
        subtitle: "Digital",
        title: "Branded",
        title2: "Mobiles",
    },
];

const Hero = () => {
    const settings = {
        dots: false,
        arrows: false,
        infinite: true,
        speed: 800,
        slidesToShow: 1,
        slidesToScroll: 1,
        autoplay: true,
        autoplaySpeed: 3000,
        cssEase: "ease-in-out",
    };

    return (
        <div className="container">
            {/* Hero Section */}
            <div className="overflow-hidden rounded-3xl min-h-[550px] sm:min-h-[650px] hero-bg-color flex justify-center items-center">
                <div className="container pb-8 sm:pb-0">
                    <Slider {...settings}>
                        {HeroData.map((data) => (
                            <div key={data.id}>
                                <div className="grid grid-cols-1 sm:grid-cols-2">
                                    {/* Text Section */}
                                    <div className="flex flex-col justify-center gap-4 sm:pl-3 pt-12 sm:pt-0 text-center sm:text-left order-2 sm:order-1 relative z-10">
                                        <h1 className="text-3xl sm:text-6xl lg:text-2xl font-bold">{data.subtitle}</h1>
                                        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold">{data.title}</h1>
                                        <h1 className="text-5xl uppercase text-white sm:text-[80px] md:text-[100px] xl:text-[150px] font-bold">{data.title2}</h1>
                                        <div>
                                            <Button
                                                text="Shop By Category"
                                                bgColor="bg-primary"
                                                textColor="text-white"
                                                onClick={() => document.getElementById('categories')?.scrollIntoView({ behavior: 'smooth' })}
                                            />
                                        </div>
                                    </div>
                                    {/* Image Section */}
                                    <div className="order-1 sm:order-2">
                                        <div>
                                            <img src={data.img} alt="" className="w-[350px] h-[350px] sm:h-[550px] sm:scale-105 lg:scale-110 object-contain mx-auto drop-shadow-[-8px_4px_6px_rgba(0,0,0,.4)] relative z-40" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </Slider>
                </div>
            </div>

            {/* Search Bar Section */}
            <SearchBar />
        </div>
    );
};

export default Hero;
