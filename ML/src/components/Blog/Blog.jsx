import React from "react";

const blogs = [
    {
        id: 1,
        title: "The Rise of Foldable Smartphones: Future or Gimmick?",
        excerpt: "Foldable phones are making a strong comeback with brands like Samsung, Oppo, and Google pushing innovation. But are they really the future of mobile technology?",
        date: "March 23, 2025"
    },
    {
        id: 2,
        title: "AI in Laptops: How Smart is Your Next Laptop?",
        excerpt: "With AI-powered processors, laptops are getting smarter. From real-time performance optimization to enhanced battery life, AI is shaping the next-gen computing experience.",
        date: "March 18, 2025"
    },
    {
        id: 3,
        title: "The E-commerce Boom: How Tech is Changing Online Shopping",
        excerpt: "With AR-based shopping, drone deliveries, and AI-powered recommendations, online shopping is evolving rapidly. What’s next for the industry?",
        date: "March 10, 2025"
    },
    {
        id: 4,
        title: "Electric Vehicles & The Future: Can EVs Dominate the Market?",
        excerpt: "Tesla, Rivian, and other brands are pushing EVs into the mainstream. With rapid charging, longer battery life, and sustainable production, is the EV revolution here to stay?",
        date: "March 5, 2025"
    }
];

const Blog = () => {
    return (
        <div id="blogs" className="py-16 px-6 sm:px-12">
            <h2 className="text-red-600 font-semibold tracking-widest text-2xl uppercase sm:text-3xl text-center">
                Latest Blogs
            </h2>
            <p className="text-gray-600 text-center mt-2">
                Explore the latest trends in smartphones, laptops, e-commerce, and future tech.
            </p>

            <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
                {blogs.map((blog) => (
                    <div key={blog.id} className="border p-5 rounded-lg shadow-md hover:shadow-lg transition">
                        <h3 className="text-xl font-semibold text-red-600">{blog.title}</h3>
                        <p className="text-gray-500 text-sm mt-1">{blog.date}</p>
                        <p className="text-gray-600 mt-3">{blog.excerpt}</p>
                        <p className="text-red-600 font-semibold mt-3 block hover:underline cursor-pointer">
                            Read More →
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Blog;