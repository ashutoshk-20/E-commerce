const productsData = [
    {
      id: 1,
      slug:"boat-rockerz-480",
      name: "Boat Rockerz 480",
      price: "₹1799",
      category: "headphones",
      image: ["/assets/B480/Head1.png", "/assets/B480/Head2.png"],
      description: [
        "Wireless Bluetooth connectivity",
        "Up to 10 hours of battery life",
        "40mm dynamic drivers for immersive sound",
        "Built-in mic for hands-free calls",
        "Ergonomic and lightweight design"
      ]
    },
    {
      id: 2,
      slug:"jbl-t-450",
      name: "JBL TODE 450BT",
      price: "₹5500",
      category: "headphones",
      image: ["/assets/J450/Head1.png", "/assets/J450/Head2.png"],
      description: [
        "Powerful JBL Pure Bass sound",
        "15 hours of playtime on a single charge",
        "Foldable, lightweight design",
        "Multi-device pairing capability",
        "Fast charging with USB-C"
      ]
    },
    {
      id: 3,
      slug:"boat-rockerz-350",
      name: "Boat Rockerz 350",
      price: "₹1199",
      category: "headphones",
      image: ["/assets/B350/Head1.png", "/assets/B350/Head2.png"],
      description: [
        "HD audio with deep bass",
        "10m Bluetooth wireless range",
        "Soft ear cushions for comfort",
        "Dual connectivity (Bluetooth & AUX)",
        "Built-in mic for voice assistants"
      ]
    },
    {
      id: 4,
      slug:"jbl-tune-750",
      name: "JBL TUNE 750BT",
      price: "₹6500",
      category: "headphones",
      image: ["/assets/J750/Head1.png", "/assets/J750/Head2.png"],
      description: [
        "Active Noise Cancellation",
        "JBL Pure Bass Sound",
        "Up to 22 hours of battery life",
        "Comfortable over-ear design",
        "Quick charging feature"
      ]
    },
    {
      id: 5,
      slug:"boat-rockerz-450",
      name: "Boat Rockerz 450 Pro",
      price: "₹1799",
      category: "headphones",
      image: ["/assets/B450P/Head1.png", "/assets/B450P/Head2.png"],
      description: [
        "Pro sound with thumping bass",
        "Up to 70 hours of battery life",
        "Bluetooth v5.0 with seamless connectivity",
        "Type-C fast charging support",
        "Lightweight and adjustable headband"
      ]
    },
    {
      id: 6,
      slug:"lenovo-yoga",
      name: "LENOVO YOGA",
      price: "₹177999",
      category: "laptops",
      image: ["/assets/LY/Lap1.png", "/assets/LY/Lap2.png"],
      description: [
        "2-in-1 convertible design",
        "12th Gen Intel Core i7 processor",
        "OLED touchscreen display",
        "Ultra-slim and lightweight",
        "Long battery life with rapid charge"
      ]
    },
    {
      id: 7,
      slug:"dell-xps",
      name: "Dell XPS 13",
      price: "₹70999",
      category: "laptops",
      image: ["/assets/DXPS/Lap1.png", "/assets/DXPS/Lap2.png"],
      description: [
        "InfinityEdge FHD display",
        "Intel Evo platform powered",
        "Lightweight aluminum chassis",
        "Fast SSD storage",
        "Windows 11 pre-installed"
      ]
    },
    {
      id: 8,
      slug:"hp-spectre",
      name: "HP Spectre x360",
      price: "₹87900",
      category: "laptops",
      image: ["/assets/HP/Lap1.png", "/assets/HP/Lap2.png"],
      description: [
        "360-degree hinge for flexible use",
        "OLED 4K touchscreen",
        "Intel i7 processor",
        "Built-in fingerprint sensor",
        "Long battery life with fast charge"
      ]
    },
    {
      id: 9,
      slug:"macbook-pro",
      name: "Apple MacBook Pro",
      price: "₹100000",
      category: "laptops",
      image: ["/assets/MP/Lap1.png", "/assets/MP/Lap2.png"],
      description: [
        "M2 chip for high performance",
        "Retina display with True Tone",
        "Magic Keyboard with Touch Bar",
        "All-day battery life",
        "Silent, fanless design"
      ]
    },
    {
      id: 10,
      slug:"macbook-air",
      name: "Apple MacBook Air M3",
      price: "₹150000",
      category: "laptops",
      image: ["/assets/MA/Lap1.png", "/assets/MA/Lap2.png"],
      description: [
        "Ultra-thin and lightweight",
        "M3 chip for smooth performance",
        "Liquid Retina display",
        "Silent fanless cooling",
        "Environmentally friendly build"
      ]
    },
    { 
        id: 11,
        slug:"iphone-16",
        name: "iPhone 16", price: "₹80000", 
        category: "iphones",
        description: [
            "Latest A-series chip for enhanced performance.",
            "High-quality dual-camera system with night mode.",
            "Improved battery life for all-day usage.",
            "iOS with regular updates and security features.",
            "5G connectivity for ultra-fast speeds."
        ],
        image: ["/assets/16/I1.png", "/assets/16/I2.png"] 
    },
    { 
        id: 12, 
        slug:"iphone-16-pro",
        name: "iPhone 16 Pro", price: "₹119000", 
        category: "iphones",
        description: [
            "ProMotion display with 120Hz refresh rate.",
            "Triple-camera system with ProRAW capabilities.",
            "Titanium build for a premium feel.",
            "A17 Pro chip for lightning-fast performance.",
            "MagSafe wireless charging and accessories support."
        ],
        image: ["/assets/16P/I1.png", "/assets/16P/I2.png"] 
    },
    { 
        id: 13, 
        slug:"iphone-15-pro",
        name: "iPhone 15 Pro", price: "₹110000", 
        category: "iphones",
        description: [
            "Titanium build with lightweight design.",
            "A17 Bionic chip for superior efficiency.",
            "ProMotion display with always-on technology.",
            "Cinematic mode for professional-grade videos.",
            "MagSafe charging and accessories support."
        ],
        image: ["/assets/15P/I1.png", "/assets/15P/I2.png"] 
    },
    { 
        id: 14,
        slug:"iphone-14-pro", 
        name: "iPhone 14 Pro", price: "₹99000", 
        category: "iphones",
        description: [
            "Always-on display with Dynamic Island feature.",
            "A16 Bionic chip for fast and smooth performance.",
            "Pro camera system with 48MP main sensor.",
            "Crash detection and emergency SOS features.",
            "All-day battery life with fast charging support."
        ],
        image: ["/assets/14P/I1.png", "/assets/14P/I2.png"] 
    },
    { 
        id: 15, 
        slug:"iphone-16e",
        name: "iPhone 16e", price: "₹59990", 
        category: "iphones",
        description: [
            "Affordable iPhone with modern features.",
            "A-series chip for smooth everyday performance.",
            "Dual-camera system with Smart HDR.",
            "iOS with long-term software updates.",
            "5G connectivity for fast internet speeds."
        ],
        image: ["/assets/16e/I1.png", "/assets/16e/I2.png"] 
    },
    { 
        id: 16,
        slug:"samsung-s25-ultra", 
        name: "Samsung S25 Ultra", price: "₹165999",
        category: "androids", 
        description: [
            "Flagship Samsung device with S-Pen support.",
            "Dynamic AMOLED display with ultra-high resolution.",
            "200MP main camera with AI-enhanced photography.",
            "Snapdragon 8 Gen chipset for top-tier performance.",
            "Long battery life with super-fast charging."
        ],
        image: ["/assets/S25U/A1.png", "/assets/S25U/A2.png"] 
    },
    { 
        id: 17, 
        slug:"samsung-s24-ultra",
        name: "Samsung S24 Ultra", price: "₹145990", 
        category: "androids",
        description: [
            "Premium Android flagship with a stunning display.",
            "High-performance chipset for smooth experience.",
            "Professional-grade camera system with AI features.",
            "Long battery life and fast charging.",
            "S-Pen support for added productivity."
        ],
        image: ["/assets/S24U/A1.png", "/assets/S24U/A2.png"] 
    },
    { 
        id: 18,
        slug:"samsung-s25",
        name: "Samsung S25", price: "₹80990", 
        category: "androids",
        description: [
            "Next-generation Samsung smartphone with AI-powered features.",
            "High-refresh-rate display for a smooth experience.",
            "5G connectivity for ultra-fast speeds.",
            "Advanced camera system for stunning photography.",
            "Premium design and build quality."
        ],
        image: ["/assets/S25/A1.png", "/assets/S25/A2.png"] 
    },
    { 
        id: 19,
        slug:"google-pixel-9-pro", 
        name: "Google Pixel 9 Pro", price: "₹120000", 
        category: "androids",
        description: [
            "Google’s flagship smartphone with AI-powered camera.",
            "Stock Android experience with Pixel-exclusive features.",
            "Ultra-fast Tensor chip for powerful performance.",
            "90Hz refresh rate display for smooth interactions.",
            "All-day battery life with adaptive charging."
        ],
        image: ["/assets/9P/A1.png", "/assets/9P/A2.png"] 
    },
    { 
        id: 20, 
        slug:"google-pixel-8a", 
        name: "Google Pixel 8a 5G", price: "₹85000", 
        category: "androids",
        description: [
            "Affordable 5G smartphone from Google.",
            "Excellent camera performance with Google’s AI enhancements.",
            "Smooth and clean Android experience.",
            "Reliable battery life with fast charging.",
            "High-quality OLED display for immersive visuals."
        ],
        image: ["/assets/8a/A1.png", "/assets/8a/A2.png"] 
    }
  ];
  
  export default productsData;
  