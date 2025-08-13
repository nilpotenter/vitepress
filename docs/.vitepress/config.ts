import { defineConfig } from 'vitepress'

// https://vitepress.vuejs.org/config/app-configs
export default defineConfig({
    // base: '/vitepress/',
    base: '/',
    title:"nilpotenter",
    head: [
        // 这将是 <link rel="icon" href="/favicon.ico">
        ['link', { rel: 'icon', href: '/assets/icons/docker_original_logo_icon_146556.ico' }]
    ],
    themeConfig:{
	
        logo: "/assets/icons/docker_original_logo_icon_146556.ico",
        sidebar:{
            "/guide/":[
                {
                    text:"简介",
                    collapsible:true,
                    items:[
                        {text:"什么是Vitepress?",link: "/guide/introduce/intro"},
                        {text:"快速开始",link: "/guide/introduce/deploy"},
                    ],
                },
                {
                    text:"写作",
                    collapsible:true,
                    items:[
                        {text:"markdown 扩展",link:"/guide/write/markdown"},
                        {text:"test",link:"/guide/write/asset-handling"},
                    ],
                },
                {
                    text:"Git",
                    collapsible:true,
                    items:[
                        {text:"git1",link:"/guide/Git/git1"},
                        {text:"git2",link:"/guide/Git/git2"},
                    ],
                },
                    
            ],
            "/reference/": [
                // 匹配/reference/路径
                {
                    text:"参考", // 分组标题
                    collapsible:true, // 可折叠
                    items: [
                        { text:"主题", link:"/reference/theme"},
                        { text:"导航栏", link:"/reference/nav"},
                        ],
                },
                        ],
            },
        //顶部菜单栏
        nav:[
            {text:"指南",link:"/guide/introduce/intro",activeMatch:"/guide/"},
            {text:"参考",link:"/reference/theme",activeMatch:"/reference/"},
            {text:"百度",link:"https://www.baidu.com/"}
        
      ]
    }
});
