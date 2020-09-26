
def parse_files(inputs):
    sentances = []
    for file in inputs:
        with open(file) as csvfile:
            lines = csvfile.readlines()
            for l in lines:
                tmp_list = []
                list_of_words = l.split(",")
                if len(list_of_words) >= 2:
                    for word in list_of_words:
                        try:
                            int(word)  # try to parse it as int, if we can ignore it, if get an error
                        except ValueError:  # then it is a normal string
                            if word == "fruit/vegetable juice":
                                word = "juice"
                            if word == "cling film/bags":
                                word = "cling film"
                            if word.__contains__("/"):
                                items = word.split("/")
                                word = items[1]
                            word = word.replace("&", "").replace("-", "_").replace(" ", "_").strip()
                            if word == "other_vegetables":
                                word = "vegetables"
                            if word == "long_life_bakery_product":
                                word = "bakery_product"
                            if word == "hamburger_meat":
                                word = "hamburger"
                            if word == "liquor_(appetizer)":
                                word = "liqour"
                            if word.__contains__("chocolate"):
                                word = "chocolate"
                            if word.__contains__("bread"):
                                word = "bread"
                            if word.__contains__("beer"):
                                word = "beer"
                            if word.__contains__("snack"):
                                word = "snack"
                            tmp_list.append(word)
                    edited_line = " ".join(tmp_list)
                    sentances.append(edited_line)
    return sentances