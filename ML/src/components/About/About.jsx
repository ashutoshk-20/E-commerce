const About = () => {
    return (
        <div id="about" className="py-16 px-6 sm:px-12">
            <h2 className="text-red-600 font-semibold tracking-widest text-2xl uppercase sm:text-3xl text-center">
                About Us
            </h2>
            <p className="text-gray-600 text-center mt-2">
                Your trusted destination for the latest and best in tech. We bring you the finest selection of smartphones, laptops, and more, ensuring quality and affordability.
            </p>

            <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border p-5 rounded-lg shadow-md hover:shadow-lg transition">
                    <h3 className="text-xl font-semibold text-red-600">Our Mission</h3>
                    <p className="text-gray-600 mt-3">
                        We aim to revolutionize the online shopping experience by providing top-notch products with seamless navigation and reliable support.
                    </p>
                </div>

                <div className="border p-5 rounded-lg shadow-md hover:shadow-lg transition">
                    <h3 className="text-xl font-semibold text-red-600">Why Choose Us?</h3>
                    <p className="text-gray-600 mt-3">
                        From the latest gadgets to exclusive deals, we ensure our customers get the best technology at the best prices with a hassle-free shopping experience.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default About;