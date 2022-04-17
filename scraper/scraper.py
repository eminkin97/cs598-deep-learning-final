import scrapy
import time

class PatientInfoSpider(scrapy.Spider):

    name = "patient_info"
    
    custom_settings = {
        'CONCURRENT_ITEMS': 1,
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 20
        
    }

    def start_requests(self):
        urlPrefix = 'https://patient.info/forums/index-'
        alphabet = 'A'
        i = 0
        while i<26:
            url = urlPrefix + chr(ord(alphabet) + i)
            yield scrapy.Request(url=url, callback=self.parse)
            i += 1

    def parse(self, response):
        if response.status == 429:
            time.sleep(30)
            
        aToZgroups = response.xpath("//div[@class='disc-forums disc-a-z']")
        symtomRowsLinks = aToZgroups.xpath(".//tr[@class='row-0']//a/@href")
        symtomRowsLinks = symtomRowsLinks[:]
        for symptomLink in symtomRowsLinks:
            symtomCompleteLink = response.urljoin(symptomLink.get())
            yield scrapy.Request(url=symtomCompleteLink, callback=self.parsePostsList)

    def parsePostsList(self, response):
        if response.status == 429:
                time.sleep(30)
                
        disease = response.xpath("//h1[@class='articleHeader__title masthead__title']/text()").get().strip()
        posts = response.xpath("//li[@class='disc-smr cardb']")
        posts = posts[:]
        for postHeading in posts:
            postLinkAnchorTag = postHeading.xpath(".//h3[@class='post__title']//a/@href")
            postLink = response.urljoin(postLinkAnchorTag.get())
            mydict = {
                'contentType': 'disease',
                'disease': disease,
                'postLink': postLink
            }
            yield mydict
            yield scrapy.Request(url=postLink, callback=self.parsePost)

    def parsePost(self, response):
        if response.status == 429:
            time.sleep(30)
            
        postLink = response.url
        postHeading = response.xpath("//h1[@class='u-h1 post__title']/text()").get()
        postContentParas = response.xpath("//div[@class='post__content']/p/text()")
        postContent = ''
        for para in postContentParas:
            postContent += para.get() + ' '
        item = {
            'contentType': 'userPost',
            'postLink': postLink,
            'postHeading': postHeading,
            'postContent': postContent
        }
        yield item